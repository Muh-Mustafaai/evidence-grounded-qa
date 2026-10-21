
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluate_scifact import DATASET_DIR


def calculate_recall(ranked_doc_ids, relevant_doc_ids, k):
    found = len(set(ranked_doc_ids[:k]) & relevant_doc_ids)
    return found / len(relevant_doc_ids) if relevant_doc_ids else 0.0


def main():
    # Load the same SciFact test data used by BM25
    corpus, queries, qrels = GenericDataLoader(
        data_folder=str(DATASET_DIR)
    ).load(split="test")

    document_ids = list(corpus.keys())
    document_texts = [
        corpus[doc_id]["title"] + " " + corpus[doc_id]["text"]
        for doc_id in document_ids
    ]

    # Load the embedding model and encode each document once
    model = SentenceTransformer("all-MiniLM-L6-v2")
    document_embeddings = model.encode(
        document_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    recalls = {1: [], 3: [], 5: []}

    # Retrieve documents for each scientific claim
    for query_id, query_text in queries.items():
        query_embedding = model.encode(
            [query_text],
            normalize_embeddings=True,
        )

        # Compare the claim embedding with all document embeddings
        scores = cosine_similarity(
            query_embedding,
            document_embeddings,
        )[0]

        ranked_indices = scores.argsort()[::-1]
        ranked_doc_ids = [
            document_ids[index] for index in ranked_indices
        ]

        relevant_doc_ids = {
            doc_id
            for doc_id, relevance in qrels.get(query_id, {}).items()
            if relevance > 0
        }

        if not relevant_doc_ids:
            continue

        for k in recalls:
            recalls[k].append(
                calculate_recall(ranked_doc_ids, relevant_doc_ids, k)
            )

    print("SciFact Semantic Retrieval Evaluation")
    print("=" * 40)
    print(f"Documents: {len(corpus)}")
    print(f"Queries loaded: {len(queries)}")
    print(f"Queries evaluated: {len(recalls[1])}")

    for k, values in recalls.items():
        average = sum(values) / len(values) if values else 0.0
        print(f"Recall@{k}: {average:.4f} ({average:.2%})")


if __name__ == "__main__":
    main()
