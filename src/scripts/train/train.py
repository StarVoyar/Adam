import math
import time
import torch
import wandb
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from torch import optim
from llm.tokenizer import BPETokenizer, TokenizerType, TokenizerConfig
from llm.transformer import Transformer, TransformerConfig
from llm.utils import fetch_device
from llm.checkpoint import save_checkpoint
from llm.dataset import create_datasets_from_sections

torch.manual_seed(42)

DATASET_PATH = "src/data/processed/unified/v001/train.txt"
TOKENIZER_PATH = "src/data/processed/unified/v001/tokenizer.json"

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
BATCH_SIZE = 8
VAL_BATCHES = 10
LOG_EVERY = 100
CKPT_EVERY = 1000
WARMUP = 1000

USE_WANDB = False
WANDB_ENTITY = "your_wandb_entity"
WANDB_PROJECT = "adam-llm"

device = fetch_device()

tokenizer = BPETokenizer(
    mapping_path=Path(TOKENIZER_CONFIG.mapping_path)
)

sections = create_datasets_from_sections(
    dataset_path=DATASET_PATH,
    context_length=MODEL_CONFIG.context_length,
    tokenizer_path=TOKENIZER_PATH,
)

Sample = Tuple[torch.Tensor, torch.Tensor]

class MergedDataset(Dataset[Sample]):
    def __init__(self, datasets: Dict[str, Any]) -> None:
        self.datasets: list[Dataset[Sample]] = []
        self.lengths: list[int] = []

        for ds in datasets.values():
            self.datasets.append(ds)
            self.lengths.append(len(ds))

        self.total: int = sum(self.lengths)

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Sample:
        for ds, length in zip(self.datasets, self.lengths):
            if idx < length:
                x, y = ds[idx]
                return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
            idx -= length

        raise IndexError(idx)

dataset = MergedDataset(sections)

train_size = int(len(dataset) * 0.9)
val_size = len(dataset) - train_size

train_ds, val_ds = torch.utils.data.random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

model = Transformer(
    vocab_size=tokenizer.vocab_size(),
    config=MODEL_CONFIG,
).to(device)

loss_fn = CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1.0,
    betas=(0.9, 0.98),
    eps=1e-9
)

def lr_schedule(step: int) -> float:
    step = max(step, 1)
    return (
        MODEL_CONFIG.embedding_dim ** -0.5
        *
        min(
            step ** -0.5,
            step * WARMUP ** -1.5
        )
    )

scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lr_schedule
)

run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

run_dir = Path("model") / run_id
run_dir.mkdir(parents=True, exist_ok=True)

wandb_run = None

if USE_WANDB:
    wandb_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_id,
        config={
            "model": MODEL_CONFIG.__dict__,
            "tokenizer": TOKENIZER_CONFIG.__dict__
        }
    )

def evaluate():
    model.eval()
    losses = []

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            loss = loss_fn(
                out.permute(0, 2, 1),
                y
            )

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

    for x, y in tqdm(
        train_loader,
        desc=f"epoch {epoch}",
        ncols=100
    ):

        x = x.to(device)
        y = y.to(device)

        out = model(x)

        loss = loss_fn(
            out.permute(0, 2, 1),
            y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        scheduler.step()

        step += 1

        if step % LOG_EVERY == 0:

            lr = optimizer.param_groups[0]["lr"]

            print(
                f"epoch={epoch} step={step} loss={loss.item():.4f} lr={lr:.2e}"
            )

            if wandb_run:
                wandb_run.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr
                    },
                    step=step
                )

        if step % CKPT_EVERY == 0:

            val_loss, ppl = evaluate()

            print(
                f"val_loss={val_loss:.4f} ppl={ppl:.2f}"
            )

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
                    "tokenizer": TOKENIZER_CONFIG.__dict__
                }
            )

    print(
        f"epoch {epoch} finished in {time.perf_counter() - start:.2f}s"
    )

loss, ppl = evaluate()

print(
    f"final loss={loss:.4f} ppl={ppl:.2f}"
)

if wandb_run:
    wandb_run.finish()
