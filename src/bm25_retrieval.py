
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# Get the main project folder so the file paths work from anywhere
ROOT_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_PATH = ROOT_DIR / "data" / "raw" / "documents.json"
QUESTIONS_PATH = ROOT_DIR / "data" / "processed" / "questions.json"


def tokenize(text: str) -> list[str]:
    # Make everything lowercase and split the text into words
    words = re.findall(r"\b\w+\b", text.lower())
    return words


def load_json(path: Path):
    # Read the JSON file and turn it into Python data
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    # Load our documents and test questions
    documents = load_json(DOCUMENTS_PATH)
    questions = load_json(QUESTIONS_PATH)

    # BM25 needs each document to be a list of words
    tokenized_documents = [
        tokenize(document["text"]) for document in documents
    ]

    # Prepare BM25 using all the documents
    bm25 = BM25Okapi(tokenized_documents)

    print("BM25 Retrieval Baseline")
    print("=" * 40)

    # Try each question and see which documents get the highest scores
    for item in questions:
        query_tokens = tokenize(item["question"])
        scores = bm25.get_scores(query_tokens)

        # Sort document positions from highest score to lowest
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        print(f"\nQuestion: {item['question']}")
        print(f"Expected relevant documents: {item['relevant_doc_ids']}")
        print("Top 3 retrieved documents:")

        # Show the three documents BM25 ranked highest
        for rank, index in enumerate(ranked_indices[:3], start=1):
            doc_id = documents[index]["doc_id"]
            score = scores[index]
            print(f"{rank}. {doc_id} (score: {score:.4f})")


if __name__ == "__main__":
    # Run the main function when this file is executed
    main()
