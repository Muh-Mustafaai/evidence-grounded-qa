
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.bm25_retrieval import load_json, tokenize
from src.hybrid_retrieval import ROOT_DIR, DOCUMENTS_PATH, QUESTIONS_PATH


def calculate_recall(ranked_doc_ids, relevant_doc_ids, k):
    # Check how many relevant documents appear in the top k
    top_k = ranked_doc_ids[:k]
    found = len(set(top_k) & set(relevant_doc_ids))

    if not relevant_doc_ids:
        return 0.0

    return found / len(relevant_doc_ids)


def evaluate_method(name, questions, rankings):
    scores_by_k = {1: [], 3: [], 5: []}

    for item, ranked_doc_ids in zip(questions, rankings):
        for k in scores_by_k:
            score = calculate_recall(
                ranked_doc_ids,
                item["relevant_doc_ids"],
                k,
            )
            scores_by_k[k].append(score)

    print(f"\n{name}")
    print("-" * 30)

    for k, scores in scores_by_k.items():
        average = sum(scores) / len(scores) if scores else 0.0
        print(f"Recall@{k}: {average:.3f} ({average:.1%})")


def main():
    # Load our shared dataset
    documents = load_json(DOCUMENTS_PATH)
    questions = load_json(QUESTIONS_PATH)
    document_texts = [doc["text"] for doc in documents]

    # Get BM25 rankings
    bm25 = BM25Okapi([tokenize(text) for text in document_texts])
    bm25_rankings = []

    for item in questions:
        scores = bm25.get_scores(tokenize(item["question"]))
        indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )
        bm25_rankings.append([documents[i]["doc_id"] for i in indices])

    # Get semantic rankings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    document_embeddings = model.encode(
        document_texts,
        normalize_embeddings=True,
    )
    semantic_rankings = []
    hybrid_rankings = []

    for item in questions:
        query_tokens = tokenize(item["question"])
        bm25_scores = bm25.get_scores(query_tokens)

        # Scale BM25 scores before combining them with semantic scores
        if bm25_scores.max() > bm25_scores.min():
            normalized_bm25 = (
                (bm25_scores - bm25_scores.min())
                / (bm25_scores.max() - bm25_scores.min())
            )
        else:
            normalized_bm25 = bm25_scores * 0

        question_embedding = model.encode(
            [item["question"]],
            normalize_embeddings=True,
        )
        semantic_scores = cosine_similarity(
            question_embedding,
            document_embeddings,
        )[0]

        semantic_indices = sorted(
            range(len(semantic_scores)),
            key=lambda i: semantic_scores[i],
            reverse=True,
        )
        semantic_rankings.append([
            documents[i]["doc_id"] for i in semantic_indices
        ])

        # Combine keyword and semantic scores with equal weights
        hybrid_scores = 0.5 * normalized_bm25 + 0.5 * semantic_scores
        hybrid_indices = sorted(
            range(len(hybrid_scores)),
            key=lambda i: hybrid_scores[i],
            reverse=True,
        )
        hybrid_rankings.append([
            documents[i]["doc_id"] for i in hybrid_indices
        ])

    print("Retrieval Evaluation Results")
    print("=" * 35)

    evaluate_method("BM25", questions, bm25_rankings)
    evaluate_method("Semantic Retrieval", questions, semantic_rankings)
    evaluate_method("Hybrid Retrieval", questions, hybrid_rankings)


if __name__ == "__main__":
    main()
