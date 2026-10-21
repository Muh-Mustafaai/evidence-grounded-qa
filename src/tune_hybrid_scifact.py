
import json
from pathlib import Path

import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.bm25_retrieval import tokenize
from src.evaluate_scifact import DATASET_DIR


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT_DIR / "results" / "metrics" / "scifact_weight_tuning.json"

BM25_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
K_VALUES = [1, 3, 5]


def calculate_recall(ranked_doc_ids, relevant_doc_ids, k):
    found = len(set(ranked_doc_ids[:k]) & relevant_doc_ids)
    return found / len(relevant_doc_ids) if relevant_doc_ids else 0.0


def main():
    # Load the training claims and their relevance labels
    corpus, queries, qrels = GenericDataLoader(
        data_folder=str(DATASET_DIR)
    ).load(split="train")

    document_ids = list(corpus.keys())
    document_texts = [
        corpus[doc_id]["title"] + " " + corpus[doc_id]["text"]
        for doc_id in document_ids
    ]

    print(f"Documents: {len(corpus)}")
    print(f"Training queries: {len(queries)}")

    # Prepare keyword retrieval once for all weight settings
    tokenized_documents = [tokenize(text) for text in document_texts]
    bm25 = BM25Okapi(tokenized_documents)

    # Encode documents once so the weight comparison is faster
    model = SentenceTransformer("all-MiniLM-L6-v2")
    document_embeddings = model.encode(
        document_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    # Store the results for each BM25 weight
    results = {}

    for bm25_weight in BM25_WEIGHTS:
        semantic_weight = 1.0 - bm25_weight
        recalls = {k: [] for k in K_VALUES}

        print(f"Evaluating BM25 weight: {bm25_weight:.2f}")

        for query_id, query_text in queries.items():
            relevant_doc_ids = {
                doc_id
                for doc_id, relevance in qrels.get(query_id, {}).items()
                if relevance > 0
            }

            # Only score queries that have at least one relevant document
            if not relevant_doc_ids:
                continue

            bm25_scores = bm25.get_scores(tokenize(query_text))
            score_range = bm25_scores.max() - bm25_scores.min()

            # Use the same min-max normalization as the earlier experiment
            if score_range > 0:
                normalized_bm25 = (
                    (bm25_scores - bm25_scores.min()) / score_range
                )
            else:
                normalized_bm25 = np.zeros_like(bm25_scores)

            query_embedding = model.encode(
                [query_text],
                normalize_embeddings=True,
            )

            semantic_scores = cosine_similarity(
                query_embedding,
                document_embeddings,
            )[0]

            hybrid_scores = (
                bm25_weight * normalized_bm25
                + semantic_weight * semantic_scores
            )

            ranked_indices = np.argsort(hybrid_scores)[::-1]
            ranked_doc_ids = [
                document_ids[index] for index in ranked_indices
            ]

            for k in K_VALUES:
                recalls[k].append(
                    calculate_recall(ranked_doc_ids, relevant_doc_ids, k)
                )

        results[str(bm25_weight)] = {
            "bm25_weight": bm25_weight,
            "semantic_weight": semantic_weight,
            "queries_evaluated": len(recalls[1]),
            "recall_at_1": float(np.mean(recalls[1])),
            "recall_at_3": float(np.mean(recalls[3])),
            "recall_at_5": float(np.mean(recalls[5])),
        }

    # Save all configurations so we can compare them later
    output = {
        "dataset": "BEIR SciFact",
        "split": "train",
        "embedding_model": "all-MiniLM-L6-v2",
        "score_normalization": "Per-query min-max normalization for BM25",
        "selection_metric": "Recall@5",
        "weight_results": results,
        "notes": (
            "Weight tuning on the training split. These results are for "
            "configuration selection, not final held-out evaluation."
        ),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nTraining Weight Comparison")
    print("=" * 65)
    print(f"{'BM25':>8} {'Semantic':>10} {'R@1':>10} {'R@3':>10} {'R@5':>10}")

    for result in results.values():
        print(
            f"{result['bm25_weight']:>8.2f} "
            f"{result['semantic_weight']:>10.2f} "
            f"{result['recall_at_1']:>10.4f} "
            f"{result['recall_at_3']:>10.4f} "
            f"{result['recall_at_5']:>10.4f}"
        )

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
