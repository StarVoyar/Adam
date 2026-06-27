from pathlib import Path
import requests
import kagglehub
import shutil

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    # "pokemon": "https://raw.githubusercontent.com/lgreski/pokemonData/master/Pokemon.csv",
    # "yugioh": "hammadus/yugioh-full-card-database-index-august-1st-2025",
    "wordnet": "dfydata/wordnet-dictionary-thesaurus-files-in-csv-format",
}

def delete_empty_folders(path: Path):
    for folder in sorted(path.rglob("*"), reverse=True):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()

def clean_folder(path: Path):
    for item in path.rglob(".complete"):
        shutil.rmtree(item)

def reset_folder(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def download_github(url: str, dest_dir: Path):
    reset_folder(dest_dir)
    r = requests.get(url)
    (dest_dir / "pokemon.csv").write_bytes(r.content)
    print(f"Saved GitHub file to {dest_dir}")

def download_kaggle(handle: str, dest_dir: Path):
    reset_folder(dest_dir)
    downloaded_path = kagglehub.dataset_download(handle, output_dir=str(dest_dir))

    for file in Path(downloaded_path).rglob("*.csv"):
        shutil.move(str(file), str(dest_dir / file.name))

    clean_folder(dest_dir)
    delete_empty_folders(dest_dir)
    print(f"Saved Kaggle dataset to {dest_dir}")

for name, source in DATASETS.items():
    print(f"\nProcessing: {name}")
    dest = DATA_DIR / name

    if source.startswith("http"):
        download_github(source, dest)
    else:
        download_kaggle(source, dest)

print("\nAll datasets downloaded.")
