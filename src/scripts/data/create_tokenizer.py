from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = ROOT / "src" / "data" / "processed" / "unified" / "all.txt"
TOKENIZER_PATH = ROOT / "src" / "data" / "processed" / "unified" / "tokenizer.json"

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=4096,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
)

tokenizer.train([str(DATASET_PATH)], trainer)
tokenizer.save(str(TOKENIZER_PATH))

print("Created tokenizer.")
