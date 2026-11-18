"""task:overfit a sequence-copy task
 
Run with:  python -m examples.copy_task
"""
"""
The model is trained to reproduce its input sequence. If every component
(masking, residuals, cross-attention, init, generator) is wired correctly,
loss drops to near zero within a few hundred steps

"""

import time
import torch
import torch.nn as nn
from transformer import make_model, subsequent_mask


def _to_4d(mask):
    """Pad a mask with dims (after the batch dim) until it is [b, *, s, s] rank:4 """
    while mask.dim() < 4:
        mask = mask.unsqueeze(1)
    return mask


class Batch:
    """One training example pair plus the masks needed for it"""

    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = _to_4d((src != pad).unsqueeze(-2))          # [b, 1, 1, src_len]
        if tgt is not None:
            self.tgt = tgt[:, :-1]                                  # decoder input
            self.tgt_y = tgt[:, 1:]                                 # expected output
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        pad_mask = _to_4d((tgt != pad).unsqueeze(-2))               # [b, 1, 1, t]
        causal = _to_4d(subsequent_mask(tgt.size(-1))).type_as(pad_mask)
        return pad_mask & causal                                    # [b, 1, t, t]


def data_gen(vocab, batch_size, nbatches, seq_len=10, device="cpu"):
    """Random integer sequences where target == source"""
    for _ in range(nbatches):
        data = torch.randint(1, vocab, size=(batch_size, seq_len), device=device)
        data[:, 0] = 1                                              # start token
        src = data.clone().detach()
        tgt = data.clone().detach()
        yield Batch(src, tgt, pad=0)


class SimpleLossCompute:
    """Apply the generator, compute NLL loss, backprop, step """

    def __init__(self, generator, criterion, optimizer=None):
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(
            x.contiguous().view(-1, x.size(-1)),
            y.contiguous().view(-1),
        ) / norm
        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return loss.item() * norm


def run_epoch(data_iter, model, loss_compute, log_every=20):
    start = time.time()
    total_tokens, total_loss, tokens = 0, 0.0, 0
    for i, batch in enumerate(data_iter):
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if log_every and i % log_every == 1:
            elapsed = time.time() - start
            print(f"  step {i:4d} | loss/token {loss / batch.ntokens:7.4f} "
                  f"| tok/s {tokens / max(elapsed, 1e-9):7.1f}")
            start, tokens = time.time(), 0
    return total_loss / total_tokens


@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    model.eval()
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src)
    for _ in range(max_len - 1):
        tgt_mask = _to_4d(subsequent_mask(ys.size(1))).type_as(src) > 0
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = model.generator(out[:, -1])
        next_word = torch.argmax(prob, dim=-1).item()
        ys = torch.cat([ys, torch.zeros(1, 1).type_as(src).fill_(next_word)], dim=1)
    return ys


def main():
    torch.manual_seed(0)
    V, SEQ_LEN = 11, 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = make_model(V, V, N=2, d_model=64, d_ff=256, h=4, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  parameters={n_params:,}\n")

    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

    for epoch in range(10):
        model.train()
        print(f"epoch {epoch}")
        run_epoch(
            data_gen(V, batch_size=32, nbatches=40, seq_len=SEQ_LEN, device=device),
            model,
            SimpleLossCompute(model.generator, criterion, optimizer),
        )

    print("\n--- greedy decode check ---")
    src = torch.randint(1, V, size=(1, SEQ_LEN), device=device)
    src[:, 0] = 1
    src_mask = torch.ones(1, 1, 1, SEQ_LEN, device=device) > 0
    out = greedy_decode(model, src, src_mask, max_len=SEQ_LEN, start_symbol=1)
    print("source :", src.tolist()[0])
    print("decoded:", out.tolist()[0])
    match = sum(int(a == b) for a, b in zip(src.tolist()[0][1:], out.tolist()[0][1:]))
    print(f"matched {match}/{SEQ_LEN - 1} positions after the start token")


if __name__ == "__main__":
    main()
