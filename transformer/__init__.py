"""A from-scratch PyTorch implementation of the Transformer (Vaswani et al., 2017)"""

from .attention import MHA, attention
from .decoder import Decoder, DecoderLayer
from .embeddings import PE, Embedding
from .encoder import Encoder, EncoderLayer
from .layers import FullyConnected, SublayerConnection, clones
from .model import EncoderDecoder, Generator, make_model, subsequent_mask

__all__ = [
    "attention",
    "MHA",
    "clones",
    "FullyConnected",
    "SublayerConnection",
    "EncoderLayer",
    "Encoder",
    "DecoderLayer",
    "Decoder",
    "Embedding",
    "PE",
    "EncoderDecoder",
    "Generator",
    "make_model",
    "subsequent_mask",
]
