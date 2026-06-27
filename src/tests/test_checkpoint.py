from pathlib import Path
from torch import nn, optim, Tensor
from src.llm.checkpoint import save_checkpoint, load_checkpoint

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(4, 4)

    def forward(self, x: Tensor) -> Tensor:
        return self.w(x)

def run():
    print("Running checkpoint test...")

    model = Tiny()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = None

    run_dir = Path("src/tests/checkpoints_test")
    ckpt_path = "test.pt"

    configs: dict[str, object] = {
        "model": {"hidden": 4},
        "data": {"dummy": True},
        "training": {"lr": 1e-3},
        "tokenizer": {"type": "bpe"},
    }

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_dir=run_dir,
        filename=ckpt_path,
        epoch=3,
        step=123,
        train_loss=0.42,
        val_loss=0.33,
        best_val_loss=0.33,
        configs=configs,
    )

    epoch, step, best, loaded_configs = load_checkpoint(
        path=run_dir / ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
    )

    print("Loaded epoch:", epoch)
    print("Loaded step:", step)
    print("Loaded best val loss:", best)
    print("Loaded configs:", loaded_configs)
