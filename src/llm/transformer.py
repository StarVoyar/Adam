from dataclasses import dataclass
import math
import torch
from torch.nn import (
    Linear,
    Module,
    ModuleList,
    Embedding,
    LayerNorm,
    ReLU,
    Dropout,
)
from torch import Tensor

@dataclass
class TransformerConfig:
    embedding_dim: int
    context_length: int
    attention_heads: int
    ff_hidden_dim: int
    n_decoders: int
    p_dropout: float

class FeedForward(Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.l1 = Linear(embedding_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, embedding_dim)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.relu(x)
        return self.l2(x)

class Decoder(Module):
    def __init__(
        self,
        embedding_dim: int,
        ff_hidden_dim: int,
        attention_heads: int,
        context_length: int,
        p_dropout: float,
        vectorized: bool = True,
    ):
        super().__init__()
        if vectorized:
            self.masked_multi_head_attention = MultiHeadAttention(
                embedding_dim=embedding_dim,
                attention_heads=attention_heads,
                context_length=context_length,
                masked=True,
            )
        else:
            self.masked_multi_head_attention = MultiHeadAttentionSeq(
                embedding_dim=embedding_dim,
                attention_heads=attention_heads,
                context_length=context_length,
                masked=True,
            )
        self.layer_norm_1 = LayerNorm(embedding_dim)
        self.layer_norm_2 = LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim=embedding_dim, hidden_dim=ff_hidden_dim)
        self.dropout = Dropout(p=p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(self.masked_multi_head_attention(x)) + x
        x = self.layer_norm_1(x)
        x = self.dropout(self.ff(x)) + x
        return self.layer_norm_2(x)

class Attention(Module):
    mask: Tensor

    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
        context_length: int,
        masked: bool,
    ):
        super().__init__()
        self.masked = masked
        self.head_dim = head_dim
        self.wq = Linear(embedding_dim, head_dim, bias=False)
        self.wk = Linear(embedding_dim, head_dim, bias=False)
        self.wv = Linear(embedding_dim, head_dim, bias=False)

        mask = torch.ones(context_length, context_length, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q_kt = q @ k.transpose(1, 2)
        q_kt = q_kt / math.sqrt(self.head_dim)
        if self.masked:
            T = x.shape[1]
            q_kt = q_kt.masked_fill(self.mask[:T, :T], float("-inf"))
        q_kt = q_kt.softmax(dim=-1)
        return q_kt @ v

class MultiHeadAttentionSeq(Module):
    def __init__(
        self,
        embedding_dim: int,
        attention_heads: int,
        context_length: int,
        masked: bool,
    ):
        super().__init__()
        assert embedding_dim % attention_heads == 0
        self.heads = ModuleList(
            [
                Attention(
                    embedding_dim=embedding_dim,
                    head_dim=embedding_dim // attention_heads,
                    context_length=context_length,
                    masked=masked,
                )
                for _ in range(attention_heads)
            ]
        )
        self.wo = Linear(embedding_dim, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x_heads = [head(x) for head in self.heads]
        x = torch.cat(tensors=x_heads, dim=-1)
        return self.wo(x)

class MultiHeadAttention(Module):
    mask: Tensor

    def __init__(
        self,
        embedding_dim: int,
        attention_heads: int,
        context_length: int,
        masked: bool,
    ):
        super().__init__()
        assert embedding_dim % attention_heads == 0
        self.attention_heads = attention_heads
        self.head_dim = embedding_dim // attention_heads
        self.wq = Linear(embedding_dim, embedding_dim, bias=False)
        self.wk = Linear(embedding_dim, embedding_dim, bias=False)
        self.wv = Linear(embedding_dim, embedding_dim, bias=False)

        self.masked = masked
        mask = torch.ones(context_length, context_length, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask)
        self.wo = Linear(embedding_dim, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.reshape(B, T, self.attention_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.attention_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.attention_heads, self.head_dim).transpose(1, 2)

        q_kt = q @ k.transpose(-2, -1)
        q_kt = q_kt / math.sqrt(self.head_dim)
        if self.masked:
            q_kt = q_kt.masked_fill(self.mask[:T, :T], float("-inf"))
        q_kt = q_kt.softmax(dim=-1)

        v_q_kt = q_kt @ v
        v_q_kt = v_q_kt.transpose(1, 2).reshape(B, T, C)
        return self.wo(v_q_kt)

class PosEncoding(Module):
    pos_encoding: Tensor

    def __init__(self, context_length: int, embedding_dim: int, p_dropout: float):
        super().__init__()
        self.dropout = Dropout(p=p_dropout)
        pos_encoding = torch.zeros(context_length, embedding_dim)

        for pos in range(context_length):
            for i in range(embedding_dim):
                angle = pos / (10000 ** ((2 * (i // 2)) / embedding_dim))
                pos_encoding[pos, i] = math.sin(angle) if i % 2 == 0 else math.cos(angle)

        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] <= self.pos_encoding.shape[0]
        return self.dropout(x + self.pos_encoding[: x.shape[1], :])

class Transformer(Module):
    def __init__(self, vocab_size: int, config: TransformerConfig):
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=config.embedding_dim
        )
        self.pos_encoding = PosEncoding(
            context_length=config.context_length,
            embedding_dim=config.embedding_dim,
            p_dropout=config.p_dropout,
        )
        self.decoders = ModuleList(
            [
                Decoder(
                    embedding_dim=config.embedding_dim,
                    attention_heads=config.attention_heads,
                    context_length=config.context_length,
                    ff_hidden_dim=config.ff_hidden_dim,
                    p_dropout=config.p_dropout,
                )
                for _ in range(config.n_decoders)
            ]
        )
        self.linear = Linear(
            in_features=config.embedding_dim, out_features=vocab_size
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        for decoder in self.decoders:
            x = decoder(x)
        logits = self.linear(x)
        return logits
