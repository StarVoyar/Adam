import torch
from pathlib import Path
from src.llm.tokenizer import BPETokenizer
from src.llm.dataset import create_datasets_from_sections
from src.llm.transformer import Transformer, TransformerConfig

def run():
    tokenizer = BPETokenizer(Path("src/data/processed/unified/tokenizer.json"))

    datasets = create_datasets_from_sections(
        "src/data/processed/unified/train.txt",
        128,
        "src/data/processed/unified/tokenizer.json"
    )

    name, ds = list(datasets.items())[0]
    x, _ = ds[0]

    x = torch.tensor(x, dtype=torch.long).unsqueeze(0)

    config = TransformerConfig(
        embedding_dim=128,
        context_length=128,
        attention_heads=4,
        ff_hidden_dim=512,
        n_decoders=2,
        p_dropout=0.1,
    )

    model = Transformer(vocab_size=tokenizer.vocab_size(), config=config)
    logits = model(x)

    print("End-to-end test:")
    print("Dataset:", name)
    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)
