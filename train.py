import math
import time

from torch import nn, optim
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

from data import *
from models.model.hierarchical_transformer import HierarchicalTransformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from util.checkpoints import save_best_models, get_best_models


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    # if hasattr(m, 'weight') and m.weight.dim() > 1:
    # nn.init.kaiming_uniform_(m.weight.data)  # He initialization
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


model = HierarchicalTransformer(
    pad_idx=pad_token_id,
    image_size=image_size,
    image_patch_size=image_patch_size,
    max_frames=max_frames,
    frame_patch_size=frame_patch_size,
    dim=dim,
    ffn_hidden_ratio=ffn_hidden_ratio,
    n_head=n_heads,
    drop_prob=drop_prob,
    max_len=max_output,
    enc_layers=enc_layers,
    dec_layers=dec_layers,
    dec_voc_size=dec_voc_size,
    device=device,
)


print(f"The model has {count_parameters(model):,} trainable parameters")
model.apply(initialize_weights)

# Load the pre-trained model weights
if pretrained:
    pretrained_state_dict = torch.load(pretrained_model, map_location=device)
    model_state_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_state_dict.items()
        if k in model_state_dict and v.size() == model_state_dict[k].size()
    }
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)

optimizer = Adam(
    params=model.parameters(), lr=init_lr, weight_decay=weight_decay, betas=betas
)

# Define the LinearLR scheduler for a warm-up period, e.g., first 5 epochs
linear_scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, total_iters=warmup
)

# Define the CosineAnnealingLR scheduler for the rest of the epochs
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=T_0, eta_min=end_lr
)

# Combine them with SequentialLR
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[warmup]
)

criterion = nn.CrossEntropyLoss(
    ignore_index=pad_token_id, label_smoothing=label_smoothing
)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(iterator):
        src = x.to(device)
        trg = y.to(device)

        optimizer.zero_grad()

        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update the gradients
        optimizer.step()

        epoch_loss += loss.item()
        if i % 200 == 0:  # Adjust the frequency as needed
            print(f"step: {round((i / len(iterator)) * 100, 2)}% , loss: {loss.item()}")

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu_1, batch_bleu_2, batch_bleu_3, batch_bleu = [], [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(iterator):
            src = x.to(device)
            trg = y.to(device)

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu_1, total_bleu_2, total_bleu_3, total_bleu = [], [], [], []

            try:
                trg_words = idx_to_word(y, vocabulary)
                # t5 tokenizer includes '▁'
                trg_words = [[item.replace("▁", " ")] for item in trg_words]
                # print('trg_words:', trg_words)

                output_idx = output.max(dim=2)[1]
                output_words = idx_to_word(output_idx, vocabulary)
                # t5 tokenizer includes '▁'
                output_words = [item.replace("▁", " ") for item in output_words]
                # print('output_words:', output_words)

                results = sacrebleu.compute(
                    predictions=output_words,
                    references=trg_words,
                    tokenize="13a",
                    smooth_method="exp",
                )

                bleu_1, bleu_2, bleu_3, bleu = get_bleu(
                    results["bp"], results["precisions"]
                )

                total_bleu_1.append(bleu_1)
                total_bleu_2.append(bleu_2)
                total_bleu_3.append(bleu_3)
                total_bleu.append(bleu)

            except Exception as e:
                print(f"Error calculating BLEU for batch {i}, item {e}")
                pass

            total_bleu_1 = (
                sum(total_bleu_1) / len(total_bleu_1) if total_bleu_1 else 0.0
            )
            total_bleu_2 = (
                sum(total_bleu_2) / len(total_bleu_2) if total_bleu_2 else 0.0
            )
            total_bleu_3 = (
                sum(total_bleu_3) / len(total_bleu_3) if total_bleu_3 else 0.0
            )
            total_bleu = sum(total_bleu) / len(total_bleu) if total_bleu else 0.0

            batch_bleu_1.append(total_bleu_1)
            batch_bleu_2.append(total_bleu_2)
            batch_bleu_3.append(total_bleu_3)
            batch_bleu.append(total_bleu)

            if i % 25 == 0:
                print(
                    "step :",
                    round((i / len(iterator)) * 100, 2),
                    "% , loss :",
                    loss.item(),
                )

    batch_bleu_1 = sum(batch_bleu_1) / len(batch_bleu_1) if batch_bleu_1 else 0.0
    batch_bleu_2 = sum(batch_bleu_2) / len(batch_bleu_2) if batch_bleu_2 else 0.0
    batch_bleu_3 = sum(batch_bleu_3) / len(batch_bleu_3) if batch_bleu_3 else 0.0
    batch_bleu = sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0.0
    return (
        epoch_loss / len(iterator),
        batch_bleu_1,
        batch_bleu_2,
        batch_bleu_3,
        batch_bleu,
    )


def run(total_epoch, best_loss):
    train_val_loss, bleus = [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu_1, bleu_2, bleu_3, bleu = evaluate(
            model, valid_iter, criterion
        )
        end_time = time.time()

        train_val_loss.append(f"{step},{train_loss},{valid_loss}\n")
        bleus.append(f"{step},{bleu_1},{bleu_2},{bleu_3},{bleu}\n")
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        scheduler.step()

        # save the best models
        save_best_models(model, bleu, step, save_dir="result", max_models=2)

        f = open("result/train_val_loss.txt", "w")
        f.write(str(train_val_loss))
        f.close()

        f = open("result/bleus.txt", "w")
        f.write(str(bleus))
        f.close()

        print(f"Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
        )
        print(f"\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}")
        print(f"\tBLEU-1 Score: {bleu_1:.3f} |  BLEU-2 Score: {bleu_2:.3f}")
        print(f"\tBLEU-3 Score: {bleu_3:.3f} |  BLEU Score: {bleu:.3f}")

    # test the final result
    best_model = get_best_models(save_dir="result")
    model.load_state_dict(torch.load(best_model))

    test_loss, bleu_1, bleu_2, bleu_3, bleu = evaluate(model, test_iter, criterion)
    test_result = f"Test Loss: {test_loss}\nbleu-1: {bleu_1}\nbleu-2: {bleu_2}\nbleu-3: {bleu_3}\nbleu: {bleu}"

    test_loss, bleu_1, bleu_2, bleu_3, bleu = evaluate(model, valid_iter, criterion)
    valid_result = f"Valid Loss: {test_loss}\nbleu-1: {bleu_1}\nbleu-2: {bleu_2}\nbleu-3: {bleu_3}\nbleu: {bleu}"
    f = open("result/test_valid_result.txt", "w")
    f.write(f"Test Result\n{test_result}\n\nValid Result\n{valid_result}")
    f.close()


if __name__ == "__main__":
    run(total_epoch=epoch, best_loss=inf)
