from pathlib import Path
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class SectionDataset(Dataset[Tuple[List[int], List[int]]]):
    def __init__(self, text: str, context_length: int, tokenizer: Tokenizer):
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.token_ids = tokenizer.encode(text).ids

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        x = self.token_ids[index:index + self.context_length]
        y = self.token_ids[index + 1:index + 1 + self.context_length]
        return x, y

def create_datasets_from_sections(
    dataset_path: str,
    context_length: int,
    tokenizer_path: str
) -> Dict[str, SectionDataset]:

    tokenizer = Tokenizer.from_file(tokenizer_path)
    text = Path(dataset_path).read_text(encoding="utf-8")

    sections: Dict[str, str] = {}
    current_source = ""
    current_body: List[str] = []

    for line in text.splitlines():
        if "-----" in line:
            if current_source and current_body:
                sections[current_source] = "\n".join(current_body)
            current_source = line.split()[0].lower()
            current_body = []
        else:
            current_body.append(line)

    if current_source and current_body:
        sections[current_source] = "\n".join(current_body)

    datasets: Dict[str, SectionDataset] = {}
    for source, body in sections.items():
        datasets[source] = SectionDataset(body, context_length, tokenizer)

    return datasets

'''
if __name__ == "__main__":
    dataset_path = "src/data/processed/unified/v001/train.txt"
    tokenizer_path = "src/data/processed/unified/v001/tokenizer.json"

    datasets = create_datasets_from_sections(dataset_path, 128, tokenizer_path)

    out_path = Path("datasets.txt")
    with out_path.open("w", encoding="utf-8") as f:
        for name, ds in datasets.items():
            f.write("\n==============================\n")
            f.write(f"Dataset: {name}\n")
            f.write("==============================\n")

            f.write(f"Token count: {len(ds.token_ids)}\n")
            f.write(f"Sample count: {len(ds)}\n")

            first_x, first_y = ds[0]
            f.write(f"First sample X: {first_x}\n")
            f.write(f"First sample Y: {first_y}\n")

            full_text = ds.tokenizer.decode(ds.token_ids)
            f.write("Full text:\n")
            f.write(full_text + "\n")
'''
