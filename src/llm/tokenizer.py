from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict
import json
import torch
from torch import tensor, Tensor
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models, pre_tokenizers, trainers, processors, decoders

class TokenizerType(Enum):
    CHAR = "char"
    BPE = "bpe"

class Tokenizer(ABC):
    @abstractmethod
    def encode(self, s: str) -> Tensor:
        pass

    @abstractmethod
    def decode(self, ix: Tensor) -> str:
        pass

    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def get_type(self) -> TokenizerType:
        pass

@dataclass
class TokenizerConfig:
    mapping_path: str
    tokenizer_type: TokenizerType

class CharTokenizer(Tokenizer):
    def __init__(self, mapping_path: Optional[Path] = None):
        self.mapping_path = mapping_path
        self.c_to_i: Dict[str, int] = {}
        self.i_to_c: Dict[int, str] = {}
        if mapping_path:
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.c_to_i = json.load(f)
                self.i_to_c = {i: c for c, i in self.c_to_i.items()}

    def build_mapping(self, input_dataset_path: Path, mapping_save_path: Path):
        text = input_dataset_path.read_text(encoding="utf-8")
        chars = sorted(set(text))
        c_to_i = {c: i for i, c in enumerate(chars)}
        with open(mapping_save_path, "w", encoding="utf-8") as f:
            json.dump(c_to_i, f, ensure_ascii=False)

    def vocab_size(self) -> int:
        return len(self.c_to_i)

    def encode(self, s: str) -> Tensor:
        return tensor([self.c_to_i[c] for c in s], dtype=torch.long)

    def decode(self, ix: Tensor) -> str:
        return "".join(self.i_to_c[int(i.item())] for i in ix)

    def get_type(self) -> TokenizerType:
        return TokenizerType.CHAR

class BPETokenizer(Tokenizer):
    def __init__(self, mapping_path: Optional[Path] = None):
        self.mapping_path = mapping_path
        if mapping_path:
            self.tokenizer = HFTokenizer.from_file(str(mapping_path))
        else:
            self.tokenizer = HFTokenizer(models.BPE())

    def build_mapping(self, input_dataset_path: Path, mapping_save_path: Path, vocab_size: int):
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        self.tokenizer.train([str(input_dataset_path)], trainer=trainer)
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.save(str(mapping_save_path))

    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, s: str) -> Tensor:
        return tensor(self.tokenizer.encode(s).ids, dtype=torch.long)

    def decode(self, ix: Tensor) -> str:
        return self.tokenizer.decode(ix.tolist(), skip_special_tokens=False)

    def get_type(self) -> TokenizerType:
        return TokenizerType.BPE

def build_human_bpe_tokenizer(input_dataset_path: Path, mapping_save_path: Path, vocab_size: int) -> BPETokenizer:
    tok = BPETokenizer()
    tok.build_mapping(input_dataset_path=input_dataset_path, mapping_save_path=mapping_save_path, vocab_size=vocab_size)
    return tok

def build_token_id_mapping(old_mapping_path: Path, new_mapping_path: Path, mapping_save_path: Path):
    old_tok = BPETokenizer(old_mapping_path)
    new_tok = BPETokenizer(new_mapping_path)
    mapping: Dict[int, list[int]] = {}
    for i in range(new_tok.vocab_size()):
        token = new_tok.tokenizer.id_to_token(i)
        old_ids = old_tok.encode(token)
        mapping[i] = old_ids.tolist()
    with open(mapping_save_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)

def remap_embeddings(checkpoint_path: Path, mapping_path: Path, output_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    old_emb = state["embedding.weight"]
    mapping = json.load(open(mapping_path, "r", encoding="utf-8"))
    vocab_size = len(mapping)
    embedding_dim = old_emb.shape[1]
    new_emb = torch.zeros(vocab_size, embedding_dim)
    for new_id_str, old_ids in mapping.items():
        new_id = int(new_id_str)
        ids = torch.tensor(old_ids, dtype=torch.long)
        vecs = old_emb[ids]
        new_emb[new_id] = vecs.mean(dim=0)
    torch.save(new_emb, output_path)
