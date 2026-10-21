
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.bm25_retrieval import tokenize
from src.evaluate_scifact import DATASET_DIR


def calculate_recall(ranked_doc_ids, relevant_doc_ids, k):
    found = len(set(ranked_doc_ids[:k]) & relevant_doc_ids)
    return found / len(relevant_doc_ids) if relevant_doc_ids else 0.0


def main():
    # Load the same SciFact test data as our other experiments
    corpus, queries, qrels = GenericDataLoader(
        data_folder=str(DATASET_DIR)
    ).load(split="test")

    document_ids = list(corpus.keys())
    document_texts = [
        corpus[doc_id]["title"] + " " + corpus[doc_id]["text"]
        for doc_id in document_ids
    ]

    # Prepare BM25 and encode all documents for semantic retrieval
    tokenized_documents = [
        tokenize(text) for text in document_texts
    ]
    bm25 = BM25Okapi(tokenized_documents)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    document_embeddings = model.encode(
        document_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    # Start with equal weights for the two retrieval methods
    bm25_weight = 0.5
    semantic_weight = 0.5
    recalls = {1: [], 3: [], 5: []}

    for query_id, query_text in queries.items():
        # Get keyword scores and scale them to the range 0 to 1
        bm25_scores = bm25.get_scores(tokenize(query_text))
        score_range = bm25_scores.max() - bm25_scores.min()

        if score_range > 0:
            normalized_bm25 = (
                (bm25_scores - bm25_scores.min()) / score_range
            )
        else:
            normalized_bm25 = np.zeros_like(bm25_scores)

        # Get semantic similarity scores for the same claim
        query_embedding = model.encode(
            [query_text],
            normalize_embeddings=True,
        )
        semantic_scores = cosine_similarity(
            query_embedding,
            document_embeddings,
        )[0]

        # Combine the scores into one ranking
        hybrid_scores = (
            bm25_weight * normalized_bm25
            + semantic_weight * semantic_scores
        )

        ranked_indices = np.argsort(hybrid_scores)[::-1]
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

    print("SciFact Hybrid Retrieval Evaluation")
    print("=" * 40)
    print(f"Documents: {len(corpus)}")
    print(f"Queries loaded: {len(queries)}")
    print(f"Queries evaluated: {len(recalls[1])}")
    print(f"BM25 weight: {bm25_weight}")
    print(f"Semantic weight: {semantic_weight}")

    for k, values in recalls.items():
        average = sum(values) / len(values) if values else 0.0
        print(f"Recall@{k}: {average:.4f} ({average:.2%})")


if __name__ == "__main__":
    main()
