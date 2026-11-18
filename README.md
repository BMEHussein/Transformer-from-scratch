[![tests](https://github.com/BMEHussein/Transformer-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/BMEHussein/Transformer-from-scratch/actions/workflows/tests.yml)

# Transformer From Scratch

A PyTorch reimplementation of the Transformer architecture from
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017),
written from the equations in the paper. 

## What's implemented

- Scaled dot-product attention with masking
- Multi-head attention
- Position-wise feed-forward network
- Residual + LayerNorm sublayer connections (changed to pre-norm instead of native post-norm for stability; see [*The Annotated Transformer*](https://nlp.seas.harvard.edu/2018/04/03/attention.html))
- Encoder layer and N-deep encoder stack
- Decoder layer (masked self-attention, cross-attention, FFN) and decoder stack
- Token embedding scaling
- Sinusoidal positional encoding (stored via `register_buffer`)
- Full encoder-decoder model, generator head, and `make_model` factory with Xavier init

## Testing

 - Identical keys: output equals the mean of the values 
 - Verify layer independence in `clones()` deepcopies, so the N stacked layers do not share weights 
 - Positional encoding: Reordered tokens produce different encoder outputs 
 - Copy task: Test the model's ability to reproduce the input sequence (n=9) as output 

Copy-task result: 9/9 positions matched ☑️

```bash
pytest tests/                # component tests
python -m examples.copy_task # end-to-end training 
```

## Usage

```python
import torch
from transformer import make_model, subsequent_mask

model = make_model(src_vocab=11, tgt_vocab=11, N=2, d_model=64, d_ff=256, h=4)
src = torch.randint(1, 11, (2, 10))
tgt = torch.randint(1, 11, (2, 8))
out = model(src, tgt, src_mask=torch.ones(2, 1, 1, 10) > 0,
            tgt_mask=subsequent_mask(8).unsqueeze(1))
logits = model.generator(out)   # [2, 8, 11] log-probabilities
```

## Installation

```bash
git clone https://github.com/BMEHussein/Transformer-from-scratch.git
cd Transformer-from-scratch
pip install -r requirements.txt
```


## Reference

Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. <https://arxiv.org/abs/1706.03762>
