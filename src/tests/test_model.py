import torch
from pathlib import Path
from llm.transformer import Transformer, TransformerConfig
from llm.tokenizer import BPETokenizer
from torch import optim

device = "cuda" if torch.cuda.is_available() else "cpu"

root = Path(__file__).resolve().parents[2]
model_root = root / "model"
latest = sorted(model_root.iterdir())[-1]
ckpt_path = latest / "adam.pt"

tokenizer_path = root / "src/data/processed/unified/tokenizer.json"
tokenizer = BPETokenizer(tokenizer_path)

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state = ckpt["model_state"]

fixed = {}
for k, v in state.items():
    if k.startswith("_orig_mod."):
        fixed[k[len("_orig_mod."):]] = v
    else:
        fixed[k] = v

vocab_size = fixed["embedding.weight"].shape[0]
embedding_dim = fixed["embedding.weight"].shape[1]
context_length = fixed["pos_encoding.pos_encoding"].shape[0]
ff_hidden_dim = fixed["decoders.0.ff.l1.weight"].shape[0]
attention_heads = fixed["decoders.0.masked_multi_head_attention.wq.weight"].shape[0] // embedding_dim
n_decoders = sum(1 for k in fixed if k.startswith("decoders.") and k.endswith(".ff.l1.weight"))

config = TransformerConfig(
    embedding_dim=embedding_dim,
    context_length=context_length,
    attention_heads=attention_heads,
    ff_hidden_dim=ff_hidden_dim,
    n_decoders=n_decoders,
    p_dropout=0.1,
)

model = Transformer(vocab_size, config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

model.load_state_dict(fixed, strict=True)
optimizer.load_state_dict(ckpt["optimizer_state"])

model.eval()

def generate(prompt: str, max_new_tokens: int = 64) -> str:
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        if x.shape[1] >= context_length:
            break
        logits = model(x)
        probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)
    return tokenizer.decode(x[0].cpu())

print(generate("happy"))
