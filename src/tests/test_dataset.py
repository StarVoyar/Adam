from src.llm.dataset import create_datasets_from_sections

def run():
    dataset_path = "src/data/processed/unified/train.txt"
    tokenizer_path = "src/data/processed/unified/tokenizer.json"

    datasets = create_datasets_from_sections(dataset_path, 128, tokenizer_path)

    print("Datasets found:", list(datasets.keys()))

    for name, ds in datasets.items():
        print(f"\n{name} dataset:")
        print("Length:", len(ds))
        x, y = ds[0]
        print("First sample X:", x[:20])
        print("First sample Y:", y[:20])
