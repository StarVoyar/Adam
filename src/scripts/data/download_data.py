from pathlib import Path
import requests
import shutil

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "pokemon": "https://raw.githubusercontent.com/lgreski/pokemonData/master/Pokemon.csv",
    # "yugioh": "hammadus/yugioh-full-card-database-index-august-1st-2025",
    # "wordnet": "dfydata/wordnet-dictionary-thesaurus-files-in-csv-format",
}

def reset_folder(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def safe_filename(url: str) -> str:
    name = url.split("/")[-1].strip()
    if not name or "." not in name:
        return "failed.csv"
    return name

def download(url: str, dest_dir: Path):
    reset_folder(dest_dir)
    r = requests.get(url)
    name = safe_filename(url)
    (dest_dir / name).write_bytes(r.content)
    print(f"Saved file to {dest_dir}")

for name, source in DATASETS.items():
    print(f"\nProcessing: {name}")
    dest = DATA_DIR / name
    download(source, dest)

print("\nAll datasets downloaded.")
