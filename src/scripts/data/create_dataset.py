from pathlib import Path
import pandas as pd
from pandas import Series
import json
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "src" / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "unified"
VAL_SIZE = 0.1

csv_files = [p for p in RAW_DATA_DIR.rglob("*.csv") if p.is_file()]
if not csv_files:
    raise FileNotFoundError(f"No CSV datasets found in {RAW_DATA_DIR}")

def convert_to_text(df: pd.DataFrame, src: str):
    df.columns = df.columns.str.lower()
    rows = []
    for _, row in df.iterrows():
        parts = []
        for k, v in row.items():
            if isinstance(v, str) and v.strip():
                parts.append(f"{k}: {v}")
        if parts:
            rows.append({"text": "\n".join(parts), "__source": src})
    if not rows:
        return None
    return pd.DataFrame(rows)

datasets = []
for path in csv_files:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    src = path.parent.name
    out = convert_to_text(df, src)
    if out is not None:
        datasets.append(out)

if not datasets:
    raise FileNotFoundError("No usable datasets found.")

dataset = pd.concat(datasets, ignore_index=True)
dataset["text"] = dataset["text"].fillna("").astype(str)

train_df, val_df = train_test_split(dataset, random_state=42, test_size=VAL_SIZE)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def serialize_row(row: Series) -> str:
    t = row["text"]
    if not isinstance(t, str):
        return ""
    return t.lower() + "\n\n"

def header(source: str) -> str:
    return f"{source.upper()} " + "-" * 56 + "\n"

def group_rows(df: pd.DataFrame):
    groups = {}
    for _, row in df.iterrows():
        src = row["__source"]
        groups.setdefault(src, []).append(serialize_row(row))
    return groups

train_groups = group_rows(train_df)
val_groups = group_rows(val_df)

all_buf = []
train_buf = []
val_buf = []

for src, entries in train_groups.items():
    h = header(src)
    all_buf.append(h)
    train_buf.append(h)
    for e in entries:
        if e.strip():
            all_buf.append(e)
            train_buf.append(e)

for src, entries in val_groups.items():
    h = header(src)
    all_buf.append(h)
    val_buf.append(h)
    for e in entries:
        if e.strip():
            all_buf.append(e)
            val_buf.append(e)

(PROCESSED_DATA_DIR / "all.txt").write_text("".join(all_buf), encoding="utf-8")
(PROCESSED_DATA_DIR / "train.txt").write_text("".join(train_buf), encoding="utf-8")
(PROCESSED_DATA_DIR / "val.txt").write_text("".join(val_buf), encoding="utf-8")

with open(PROCESSED_DATA_DIR / "metadata.json", "w") as meta_f:
    json.dump(
        {
            "sources": [str(p) for p in csv_files],
            "val_size": VAL_SIZE,
            "n_train": sum(len(v) for v in train_groups.values()),
            "n_val": sum(len(v) for v in val_groups.values()),
            "total": sum(len(v) for v in train_groups.values()) + sum(len(v) for v in val_groups.values()),
        },
        meta_f,
        indent=2,
    )

print("Unified dataset created with grouped headers.")
