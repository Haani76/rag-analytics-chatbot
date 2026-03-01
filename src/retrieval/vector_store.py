import os
import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from configs.config import config


class VectorStore:
    """
    Vector store using FAISS and sentence-transformers.
    No ChromaDB dependency — simpler and more reliable.
    """

    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.dimension = 384  # all-MiniLM-L6-v2 output size
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine similarity
        self.documents = []
        self.store_path = os.path.join(config.BASE_DIR, "data", "embeddings", "faiss")
        os.makedirs(self.store_path, exist_ok=True)
        print(f"Vector store initialized")
        print(f"Embedding model: {config.EMBEDDING_MODEL}")

    def embed_text(self, text: str) -> np.ndarray:
        embedding = self.embedding_model.encode([text], normalize_embeddings=True)
        return embedding.astype(np.float32)

    def add_documents(self, documents: list) -> int:
        if not documents:
            return 0
        contents = [doc["content"] for doc in documents]
        print(f"Embedding {len(documents)} documents...")
        embeddings = self.embedding_model.encode(contents, normalize_embeddings=True).astype(np.float32)
        self.index.add(embeddings)
        self.documents.extend(documents)
        self.save()
        print(f"Added {len(documents)} documents. Total: {len(self.documents)}")
        return len(documents)

    def search(self, query: str, top_k: int = None) -> list:
        top_k = top_k or config.TOP_K_RESULTS
        if len(self.documents) == 0:
            return []
        query_embedding = self.embed_text(query)
        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= config.SIMILARITY_THRESHOLD:
                doc = self.documents[idx]
                results.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "title": doc["title"],
                    "category": doc["category"],
                    "similarity": round(float(score), 4),
                })
        return results

    def save(self):
        faiss.write_index(self.index, os.path.join(self.store_path, "index.faiss"))
        with open(os.path.join(self.store_path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self) -> bool:
        index_path = os.path.join(self.store_path, "index.faiss")
        docs_path = os.path.join(self.store_path, "documents.pkl")
        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            print(f"Loaded {len(self.documents)} documents from disk.")
            return True
        return False

    def get_all_documents(self) -> list:
        return self.documents


def build_vector_store() -> VectorStore:
    docs_path = os.path.join(config.PROCESSED_DATA_DIR, "kpi_documents.json")
    with open(docs_path, "r") as f:
        documents = json.load(f)

    store = VectorStore()
    if not store.load():
        store.add_documents(documents)
    return store


if __name__ == "__main__":
    print("Building vector store with FAISS...")
    store = build_vector_store()

    test_queries = [
        "What is the total revenue?",
        "Which region performs best?",
        "What is the product utilization rate?",
        "Show me revenue trends",
        "How are enterprise customers performing?",
    ]

    print("\n--- Testing semantic search ---")
    for query in test_queries:
        results = store.search(query, top_k=2)
        print(f"\nQuery: '{query}'")
        for r in results:
            print(f"  [{r['similarity']:.3f}] {r['title']}")