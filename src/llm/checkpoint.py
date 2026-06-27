from pathlib import Path
import torch
from torch import optim
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler

def save_checkpoint(
    model: Module,
    optimizer: optim.Optimizer,
    scheduler: LRScheduler | None,
    run_dir: Path,
    filename: str,
    epoch: int,
    step: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    configs: dict[str, object],
):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / filename
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "configs": configs,
    }
    torch.save(payload, path)
    print(f"Saved checkpoint: {path}")

def load_checkpoint(
    path: Path,
    model: Module,
    optimizer: optim.Optimizer,
    scheduler: LRScheduler | None,
    device: str,
):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt["scheduler_state"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["step"], ckpt["best_val_loss"], ckpt["configs"]
