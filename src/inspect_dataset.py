
from pathlib import Path
import zipfile

from beir.datasets.data_loader import GenericDataLoader


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
ZIP_PATH = RAW_DIR / "scifact.zip"


def main():
    # Extract the dataset if it hasn't been extracted yet
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(RAW_DIR)

    # Find the folder that contains the corpus file
    corpus_files = list(RAW_DIR.rglob("corpus.jsonl"))

    if not corpus_files:
        raise FileNotFoundError(
            "Could not find corpus.jsonl after extracting SciFact."
        )

    dataset_folder = corpus_files[0].parent
    print(f"Dataset folder: {dataset_folder}")

    # Load the test queries and their relevance judgments
    corpus, queries, qrels = GenericDataLoader(
        data_folder=str(dataset_folder)
    ).load(split="test")

    print("\nSciFact Test Dataset")
    print("=" * 40)
    print(f"Documents: {len(corpus)}")
    print(f"Queries: {len(queries)}")
    print(f"Queries with relevance judgments: {len(qrels)}")

    print("\nExample document:")
    print(next(iter(corpus.items())))

    print("\nExample query:")
    print(next(iter(queries.items())))

    print("\nExample relevance judgments:")
    print(next(iter(qrels.items())))


if __name__ == "__main__":
    main()
