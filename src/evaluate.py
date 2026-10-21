
import json
from pathlib import Path

from src.bm25_retrieval import load_json, tokenize

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = ROOT_DIR / "data" / "raw" / "documents.json"
QUESTIONS_PATH = ROOT_DIR / "data" / "processed" / "questions.json"


def calculate_recall(ranked_doc_ids, relevant_doc_ids, k):
    # Check how many of the relevant documents made it into the top k
    top_k = ranked_doc_ids[:k]
    found = len(set(top_k) & set(relevant_doc_ids))

    # Avoid division by zero if a question has no relevant documents
    if not relevant_doc_ids:
        return 0.0

    return found / len(relevant_doc_ids)


def main():
    # Load the documents and questions we already prepared
    documents = load_json(DOCUMENTS_PATH)
    questions = load_json(QUESTIONS_PATH)

    # Keep the same BM25 setup as our retrieval script
    from rank_bm25 import BM25Okapi

    tokenized_documents = [
        tokenize(document["text"]) for document in documents
    ]
    bm25 = BM25Okapi(tokenized_documents)

    scores_by_k = {1: [], 3: [], 5: []}

    # Test each question and measure how well BM25 ranks its documents
    for item in questions:
        query_tokens = tokenize(item["question"])
        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        ranked_doc_ids = [
            documents[index]["doc_id"] for index in ranked_indices
        ]

        relevant_doc_ids = item["relevant_doc_ids"]

        for k in scores_by_k:
            recall = calculate_recall(
                ranked_doc_ids, relevant_doc_ids, k
            )
            scores_by_k[k].append(recall)

    # Average the scores across all questions
    print("BM25 Evaluation Results")
    print("=" * 35)

    for k, recalls in scores_by_k.items():
        average_recall = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"Recall@{k}: {average_recall:.3f} ({average_recall:.1%})")


if __name__ == "__main__":
    main()
