
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.bm25_retrieval import load_json


ROOT_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = ROOT_DIR / "data" / "raw" / "documents.json"
QUESTIONS_PATH = ROOT_DIR / "data" / "processed" / "questions.json"


def main():
    # Load the same documents and questions we used for BM25
    documents = load_json(DOCUMENTS_PATH)
    questions = load_json(QUESTIONS_PATH)

    document_texts = [document["text"] for document in documents]

    # This model turns each text into a vector representing its meaning
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # We only need to encode the documents once
    document_embeddings = model.encode(
        document_texts,
        normalize_embeddings=True,
    )

    print("Semantic Retrieval Baseline")
    print("=" * 40)

    for item in questions:
        # Turn the question into a vector too
        question_embedding = model.encode(
            [item["question"]],
            normalize_embeddings=True,
        )

        # Compare the question with every document
        scores = cosine_similarity(
            question_embedding,
            document_embeddings,
        )[0]

        # Put the documents in order from most similar to least similar
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        print(f"\nQuestion: {item['question']}")
        print(f"Expected relevant documents: {item['relevant_doc_ids']}")
        print("Top 3 retrieved documents:")

        for rank, index in enumerate(ranked_indices[:3], start=1):
            doc_id = documents[index]["doc_id"]
            score = scores[index]
            print(f"{rank}. {doc_id} (similarity: {score:.4f})")


if __name__ == "__main__":
    # Start the retrieval experiment
    main()
