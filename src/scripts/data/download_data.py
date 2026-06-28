from pathlib import Path
import requests
import kagglehub
import shutil

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
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

def safe_filename(url: str) -> str:
    name = url.split("/")[-1].strip()
    if "." not in name or not name:
        return "failed.csv"
    return name

def download_github(url: str, dest_dir: Path):
    reset_folder(dest_dir)
    r = requests.get(url)
    name = safe_filename(url)
    (dest_dir / name).write_bytes(r.content)
    print(f"Saved GitHub file to {dest_dir / name}")

def download_kaggle(handle: str, dest_dir: Path):
    reset_folder(dest_dir)
    downloaded_path = kagglehub.dataset_download(handle, output_dir=str(dest_dir))

    csv_files = list(Path(downloaded_path).rglob("*.csv"))
    if not csv_files:
        (dest_dir / "failed.csv").write_text("")
        print(f"No CSV found in Kaggle dataset: {handle}")
        return

    for file in csv_files:
        name = file.name if "." in file.name else "failed.csv"
        shutil.move(str(file), str(dest_dir / name))

    clean_folder(dest_dir)
    delete_empty_folders(dest_dir)
    print(f"Saved Kaggle dataset to {dest_dir} ({len(csv_files)} files)")

for name, source in DATASETS.items():
    print(f"\nProcessing: {name}")
    dest = DATA_DIR / name

    if source.startswith("http"):
        download_github(source, dest)
    else:
        download_kaggle(source, dest)

print("\nAll datasets downloaded.")
