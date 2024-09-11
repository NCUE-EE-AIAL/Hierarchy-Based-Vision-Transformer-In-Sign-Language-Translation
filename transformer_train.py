import math
import time

from torch import nn, optim
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

from data import *
from models.model.partition_transformer import PartitionTransformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from util.saved import save_best_models
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight.data, -0.1, 0.1)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
"""
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_uniform_(m.weight.data)



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
                 weight_decay=weight_decay,)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=17000, T_mult=1)

criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id,
                                label_smoothing=0.1)

# create`torch.cuda.amp.GradScaler`
# scaler = GradScaler(init_scale=2.0)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(iterator):
        src = x.to(device)
        trg = y.to(device)

        optimizer.zero_grad()

        # FP32 -> FP16
        # with autocast(enabled=False):
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)

        # Loss Scale
        # scaler.scale(loss).backward()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update the gradients
        # scaler.step(optimizer)
        optimizer.step()

        # update scaler
        # scaler.update()

        epoch_loss += loss.item()
        if i % 100 == 0:  # Adjust the frequency as needed
            print(f'step: {round((i / len(iterator)) * 100, 2)}% , loss: {loss.item()}')

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
                trg_words = [[item] for item in trg_words]
                # print('trg_words:', trg_words)

                output_idx = output.max(dim=2)[1]
                output_words = idx_to_word(output_idx, vocabulary)
                # print('output_words:', output_words)

                # smooth_fn = SmoothingFunction().method3 # method1
                bleu_1 = corpus_bleu(trg_words, output_words, weights=(1, 0, 0, 0))  # , smoothing_function=smooth_fn
                bleu_2 = corpus_bleu(trg_words, output_words, weights=(0.5, 0.5, 0, 0))
                bleu_3 = corpus_bleu(trg_words, output_words, weights=(0.33333, 0.33333, 0.33333, 0))
                bleu = corpus_bleu(trg_words, output_words, weights=(0.25, 0.25, 0.25, 0.25))

                print(f'BLEU-1 Score: {bleu_1:.3f} | BLEU-2 Score: {bleu_2:.3f} | BLEU-3 Score: {bleu_3:.3f} | BLEU Score: {bleu:.3f}')
                total_bleu_1.append(bleu_1)
                total_bleu_2.append(bleu_2)
                total_bleu_3.append(bleu_3)
                total_bleu.append(bleu)

            except Exception as e:
                print(f"Error calculating BLEU for batch {i}, item {e}")
                pass

            total_bleu_1 = sum(total_bleu_1) / len(total_bleu_1)
            total_bleu_2 = sum(total_bleu_2) / len(total_bleu_2)
            total_bleu_3 = sum(total_bleu_3) / len(total_bleu_3)
            total_bleu = sum(total_bleu) / len(total_bleu)

            batch_bleu_1.append(total_bleu_1)
            batch_bleu_2.append(total_bleu_2)
            batch_bleu_3.append(total_bleu_3)
            batch_bleu.append(total_bleu)

            if i % 50 == 0:
                print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    batch_bleu_1 = sum(batch_bleu_1) / len(batch_bleu_1) if batch_bleu_1 else 0.0
    batch_bleu_2 = sum(batch_bleu_2) / len(batch_bleu_2) if batch_bleu_2 else 0.0
    batch_bleu_3 = sum(batch_bleu_3) / len(batch_bleu_3) if batch_bleu_3 else 0.0
    batch_bleu = sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0.0
    return epoch_loss / len(iterator), batch_bleu_1, batch_bleu_2, batch_bleu_3, batch_bleu


def run(total_epoch, best_loss):
    train_losses, test_losses = [], []
    bleus_1, bleus_2, bleus_3, bleus = [], [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu_1, bleu_2, bleu_3, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus_1.append(bleu_1)
        bleus_2.append(bleu_2)
        bleus_3.append(bleu_3)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            save_best_models(model, bleu, step, save_dir='saved', max_models=3)

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu_1.txt', 'w')
        f.write(str(bleus_1))
        f.close()

        f = open('result/bleu_2.txt', 'w')
        f.write(str(bleus_2))
        f.close()

        f = open('result/bleu_3.txt', 'w')
        f.write(str(bleus_3))
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
        print(f'\tBLEU-1 Score: {bleu_1:.3f} |  BLEU-2 Score: {bleu_2:.3f}')
        print(f'\tBLEU-3 Score: {bleu_3:.3f} |  BLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
