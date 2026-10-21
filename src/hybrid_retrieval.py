
import json
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.bm25_retrieval import load_json, tokenize


ROOT_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = ROOT_DIR / "data" / "raw" / "documents.json"
QUESTIONS_PATH = ROOT_DIR / "data" / "processed" / "questions.json"


def main():
    # Load the documents and questions
    documents = load_json(DOCUMENTS_PATH)
    questions = load_json(QUESTIONS_PATH)
    document_texts = [doc["text"] for doc in documents]

    # Prepare BM25 for keyword matching
    tokenized_documents = [tokenize(text) for text in document_texts]
    bm25 = BM25Okapi(tokenized_documents)

    # Prepare the semantic model and encode the documents
    model = SentenceTransformer("all-MiniLM-L6-v2")
    document_embeddings = model.encode(
        document_texts,
        normalize_embeddings=True,
    )

    # Give both methods equal weight for our first experiment
    bm25_weight = 0.5
    semantic_weight = 0.5

    print("Hybrid Retrieval Baseline")
    print("=" * 40)

    for item in questions:
        # Get keyword scores for the question
        query_tokens = tokenize(item["question"])
        bm25_scores = bm25.get_scores(query_tokens)

        # Scale BM25 scores so they can be combined with similarity scores
        if bm25_scores.max() > bm25_scores.min():
            bm25_scores = (
                (bm25_scores - bm25_scores.min())
                / (bm25_scores.max() - bm25_scores.min())
            )
        else:
            bm25_scores = bm25_scores * 0

        # Get semantic similarity scores
        question_embedding = model.encode(
            [item["question"]],
            normalize_embeddings=True,
        )
        semantic_scores = cosine_similarity(
            question_embedding,
            document_embeddings,
        )[0]

        # Combine the two score lists
        hybrid_scores = (
            bm25_weight * bm25_scores
            + semantic_weight * semantic_scores
        )

        # Rank documents using the combined scores
        ranked_indices = sorted(
            range(len(hybrid_scores)),
            key=lambda index: hybrid_scores[index],
            reverse=True,
        )

        print(f"\nQuestion: {item['question']}")
        print(f"Expected relevant documents: {item['relevant_doc_ids']}")
        print("Top 3 retrieved documents:")

        for rank, index in enumerate(ranked_indices[:3], start=1):
            doc_id = documents[index]["doc_id"]
            score = hybrid_scores[index]
            print(f"{rank}. {doc_id} (hybrid score: {score:.4f})")


if __name__ == "__main__":
    # Run the hybrid retrieval experiment
    main()
