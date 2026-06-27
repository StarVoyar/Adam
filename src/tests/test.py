import test_tokenizer, test_dataset, test_transformer, test_checkpoint, test_end_to_end

import shutil
from pathlib import Path
import sys

if __name__ == "__main__":
    with open("src/tests/tests.txt", "w", encoding="utf-8") as f:
        sys.stdout = f

        print("=== Running Tokenizer Test ===")
        test_tokenizer.run()

        print("\n=== Running Dataset Test ===")
        test_dataset.run()

        print("\n=== Running Transformer Test ===")
        test_transformer.run()

        print("\n=== Running Checkpoint Test ===")
        test_checkpoint.run()

        print("\n=== Running End-to-End Test ===")
        test_end_to_end.run()

        print("\n=== Cleaning up __pycache__ folders ===")

        root = Path("src")
        for pycache in root.rglob("__pycache__"):
            shutil.rmtree(pycache, ignore_errors=True)
            print(f"Deleted: {pycache}")

        print("\n=== Done ===")

        sys.stdout = sys.__stdout__

        print("All tests completed. Check 'tests.txt' for details.")
