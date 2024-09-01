import math
import time

from torch import nn, optim
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

from conf import *
from data import *
from models.model.partition_transformer import PartitionTransformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from util.saved import save_best_models
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""
swin transformer
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
        
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_uniform(m.weight.data)
"""
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Embedding):
        # Normal initialization for embeddings
        nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
    elif isinstance(m, nn.LayerNorm):
        # Set biases to zero and weights to ones for LayerNorm
        nn.init.ones_(m.weight.data)
        nn.init.zeros_(m.bias.data)


model = PartitionTransformer(trg_pad_idx=pad_token_id,
                             image_size=image_size,
                             image_patch_size=image_patch_size,
                             max_frames=max_frames,
                             frame_patch_size=frame_patch_size,
                             dim=dim,
                             ffn_hidden=ffn_hidden,
                             n_head=n_heads,
                             drop_prob=drop_prob,
                             max_len=max_len,
                             dec_voc_size=dec_voc_size,
                             device=device)


print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

# create`torch.cuda.amp.GradScaler`
scaler = GradScaler()

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(iterator):
        src = x.to(device)
        trg = y.to(device)

        optimizer.zero_grad()

        # FP32 -> FP16
        with autocast():
            output = model(src, trg)
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg.contiguous().view(-1)

            loss = criterion(output_reshape, trg)

        # Loss Scale
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update the gradients
        scaler.step(optimizer)

        # update scaler
        scaler.update()

        epoch_loss += loss.item()
        if i % 10 == 0:  # Adjust the frequency as needed
            print(f'step: {round((i / len(iterator)) * 100, 2)}% , loss: {loss.item()}')

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, (x, y) in enumerate(iterator):
            src = x.to(device)
            trg = y.to(device)

            output = model(src, trg)
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg.contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(y[j], tokenizer.tokenizer)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, tokenizer.tokenizer)
                    print("trg_words : ", trg_words)
                    print("output_words : ", output_words)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())

                    total_bleu.append(bleu)
                except Exception as e:
                    print(f"Error calculating BLEU for batch {i}, item {j}: {e}")
                    pass
            if total_bleu:
                total_bleu = sum(total_bleu) / len(total_bleu)
                batch_bleu.append(total_bleu)

            print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    batch_bleu = sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0.0
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        print('train_loss :', train_loss)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            save_best_models(model, bleu, step, save_dir='saved', max_models=3)

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
