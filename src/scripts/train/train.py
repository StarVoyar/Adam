import math
import time
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Any, cast
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss, Module
from torch import optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from llm.tokenizer import BPETokenizer, TokenizerType, TokenizerConfig
from llm.transformer import Transformer, TransformerConfig
from llm.checkpoint import save_checkpoint
from llm.dataset import create_datasets_from_sections

torch.manual_seed(42)

DATASET_PATH = "src/data/processed/unified/train.txt"
TOKENIZER_PATH = "src/data/processed/unified/tokenizer.json"

TOKENIZER_CONFIG = TokenizerConfig(
    mapping_path=TOKENIZER_PATH,
    tokenizer_type=TokenizerType.BPE,
)

MODEL_CONFIG = TransformerConfig(
    embedding_dim=256,
    context_length=64,
    attention_heads=8,
    ff_hidden_dim=1024,
    n_decoders=4,
    p_dropout=0.1,
)

TRAIN_EPOCHS = 1
BATCH_SIZE = 256
VAL_BATCHES = 10
LOG_EVERY = 100
CKPT_EVERY = 1000
WARMUP = 1000
ACCUM_STEPS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"

if use_cuda:
    autocast_device = "cuda"
    scaler = GradScaler("cuda")
else:
    autocast_device = "cpu"
    scaler = GradScaler(enabled=False)

num_workers = 8 if use_cuda else 0
persistent = True if use_cuda else False
pin = True if use_cuda else False
prefetch = 4 if use_cuda else None

if use_cuda:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

tokenizer = BPETokenizer(mapping_path=Path(TOKENIZER_PATH))

sections = create_datasets_from_sections(
    dataset_path=DATASET_PATH,
    context_length=MODEL_CONFIG.context_length,
    tokenizer_path=TOKENIZER_PATH,
)

Sample = Tuple[torch.Tensor, torch.Tensor]

class MergedDataset(Dataset[Sample]):
    def __init__(self, datasets: Dict[str, Any]) -> None:
        self.datasets = list(datasets.values())
        self.lengths = [len(ds) for ds in self.datasets]
        self.total = sum(self.lengths)

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Sample:
        for ds, length in zip(self.datasets, self.lengths):
            if idx < length:
                x, y = ds[idx]
                return (
                    torch.as_tensor(x, dtype=torch.long),
                    torch.as_tensor(y, dtype=torch.long),
                )
            idx -= length
        raise IndexError(idx)

dataset = MergedDataset(sections)

train_size = int(len(dataset) * 0.9)
val_size = len(dataset) - train_size

train_ds, val_ds = torch.utils.data.random_split(
    dataset,
    [train_size, val_size],
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=pin,
    num_workers=num_workers,
    persistent_workers=persistent,
    prefetch_factor=prefetch,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=pin,
    num_workers=num_workers,
    persistent_workers=persistent,
    prefetch_factor=prefetch,
)

model: Module = Transformer(
    vocab_size=tokenizer.vocab_size(),
    config=MODEL_CONFIG,
).to(device)

if use_cuda:
    model = cast(
        Module,
        torch.compile(model, mode="max-autotune", fullgraph=True),
    )

loss_fn = CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=True,
)

def lr_schedule(step: int) -> float:
    step = max(step, 1)
    return MODEL_CONFIG.embedding_dim ** -0.5 * min(step ** -0.5, step * WARMUP ** -1.5)

scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lr_schedule,
)

run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
run_dir = Path("model") / run_id
run_dir.mkdir(parents=True, exist_ok=True)

def evaluate():
    model.eval()
    losses = []
    with torch.no_grad(), autocast(autocast_device):
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            loss = loss_fn(out.permute(0, 2, 1), y)
            losses.append(loss.item())
            if i >= VAL_BATCHES:
                break
    model.train()
    avg = sum(losses) / len(losses)
    return avg, math.exp(avg)

step = 0
best = float("inf")

for epoch in range(TRAIN_EPOCHS):
    start = time.perf_counter()
    for x, y in tqdm(train_loader, desc=f"epoch {epoch}", ncols=100):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(autocast_device):
            out = model(x)
            loss = loss_fn(out.permute(0, 2, 1), y)

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        step += 1

        if step % LOG_EVERY == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"epoch={epoch} step={step} loss={loss.item():.4f} lr={lr:.2e}")

        if step % CKPT_EVERY == 0:
            val_loss, ppl = evaluate()
            print(f"val_loss={val_loss:.4f} ppl={ppl:.2f}")

            name = "checkpoint.pt"
            if val_loss < best:
                best = val_loss
                name = "adam.pt"

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                run_dir=run_dir,
                filename=name,
                epoch=epoch,
                step=step,
                train_loss=loss.item(),
                val_loss=val_loss,
                best_val_loss=best,
                configs={
                    "model": MODEL_CONFIG.__dict__,
                    "tokenizer": TOKENIZER_CONFIG.__dict__,
                },
            )

    print(f"epoch {epoch} finished in {time.perf_counter() - start:.2f}s")

loss, ppl = evaluate()
print(f"final loss={loss:.4f} ppl={ppl:.2f}")
