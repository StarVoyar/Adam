import torch
from src.llm.transformer import Transformer, TransformerConfig

def run():
    config = TransformerConfig(
        embedding_dim=128,
        context_length=128,
        attention_heads=4,
        ff_hidden_dim=512,
        n_decoders=2,
        p_dropout=0.1,
    )

    vocab_size = 3000
    model = Transformer(vocab_size=vocab_size, config=config)

    x = torch.randint(0, vocab_size, (1, 128), dtype=torch.long)
    logits = model(x)

    print("Transformer test:")
    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)
