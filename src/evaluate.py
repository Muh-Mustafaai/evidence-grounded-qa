
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from src.bm25_retrieval import load_json, tokenize


ROOT_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = ROOT_DIR / "data" / "raw" / "documents.json"
QUESTIONS_PATH = ROOT_DIR / "data" / "processed" / "questions.json"


def calculate_recall(ranked_doc_ids, relevant_doc_ids, k):
    # Check how many relevant documents appear in the top k
    top_k = ranked_doc_ids[:k]
    found = len(set(top_k) & set(relevant_doc_ids))

    if not relevant_doc_ids:
        return 0.0

    return found / len(relevant_doc_ids)


def evaluate_method(name, documents, questions, rankings):
    scores_by_k = {1: [], 3: [], 5: []}

    for item, ranked_doc_ids in zip(questions, rankings):
        for k in scores_by_k:
            recall = calculate_recall(
                ranked_doc_ids,
                item["relevant_doc_ids"],
                k,
            )
            scores_by_k[k].append(recall)

    print(f"\n{name}")
    print("-" * 30)

    for k, recalls in scores_by_k.items():
        average = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"Recall@{k}: {average:.3f} ({average:.1%})")


def main():
    # Load the documents and questions
    documents = load_json(DOCUMENTS_PATH)
    questions = load_json(QUESTIONS_PATH)
    document_texts = [doc["text"] for doc in documents]

    # Rank documents with BM25
    tokenized_documents = [tokenize(text) for text in document_texts]
    bm25 = BM25Okapi(tokenized_documents)

    bm25_rankings = []

    for item in questions:
        scores = bm25.get_scores(tokenize(item["question"]))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )
        bm25_rankings.append([
            documents[index]["doc_id"] for index in ranked_indices
        ])

    # Rank documents by semantic similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")
    document_embeddings = model.encode(
        document_texts,
        normalize_embeddings=True,
    )

    semantic_rankings = []

    for item in questions:
        question_embedding = model.encode(
            [item["question"]],
            normalize_embeddings=True,
        )
        scores = cosine_similarity(
            question_embedding,
            document_embeddings,
        )[0]

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )
        semantic_rankings.append([
            documents[index]["doc_id"] for index in ranked_indices
        ])

    print("Retrieval Evaluation Results")
    print("=" * 35)

    evaluate_method("BM25", documents, questions, bm25_rankings)
    evaluate_method(
        "Semantic Retrieval",
        documents,
        questions,
        semantic_rankings,
    )


if __name__ == "__main__":
    main()
