from pathlib import Path
from src.llm.tokenizer import BPETokenizer

def run():
    tokenizer_path = Path("src/data/processed/unified/tokenizer.json")
    tokenizer = BPETokenizer(mapping_path=tokenizer_path)

    text = "test text"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print("Tokenizer test:")
    print("Input:", text)
    print("Encoded:", ids.tolist())
    print("Decoded:", decoded)
