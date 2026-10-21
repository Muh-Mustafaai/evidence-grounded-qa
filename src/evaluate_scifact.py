
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi

from src.bm25_retrieval import tokenize


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "data" / "raw" / "scifact"


def main():
    # Load SciFact's test documents, claims, and relevance labels
    corpus, queries, qrels = GenericDataLoader(
        data_folder=str(DATASET_DIR)
    ).load(split="test")

    # Keep document IDs alongside their text
    document_ids = list(corpus.keys())
    document_texts = [
        corpus[doc_id]["title"] + " " + corpus[doc_id]["text"]
        for doc_id in document_ids
    ]

    # Prepare BM25 using the full document collection
    tokenized_documents = [tokenize(text) for text in document_texts]
    bm25 = BM25Okapi(tokenized_documents)

    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []

    # Test each claim against the document collection
    for query_id, query_text in queries.items():
        scores = bm25.get_scores(tokenize(query_text))

        # Rank every document from highest score to lowest
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )
        ranked_doc_ids = [
            document_ids[index] for index in ranked_indices
        ]

        # Get the relevant documents for this claim
        relevant_doc_ids = {
            doc_id
            for doc_id, relevance in qrels.get(query_id, {}).items()
            if relevance > 0
        }

        # Skip claims that have no relevant documents
        if not relevant_doc_ids:
            continue

        def recall_at(k):
            found = len(set(ranked_doc_ids[:k]) & relevant_doc_ids)
            return found / len(relevant_doc_ids)

        recall_at_1.append(recall_at(1))
        recall_at_3.append(recall_at(3))
        recall_at_5.append(recall_at(5))

    # Average recall across all evaluated claims
    print("SciFact BM25 Evaluation")
    print("=" * 35)
    print(f"Documents: {len(corpus)}")
    print(f"Queries loaded: {len(queries)}")
    print(f"Queries evaluated: {len(recall_at_1)}")

    for k, values in [
        (1, recall_at_1),
        (3, recall_at_3),
        (5, recall_at_5),
    ]:
        average = sum(values) / len(values) if values else 0.0
        print(f"Recall@{k}: {average:.4f} ({average:.2%})")


if __name__ == "__main__":
    main()
