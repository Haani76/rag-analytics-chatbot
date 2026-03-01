import os
import time
import json
import datetime
from configs.config import config


class RAGPipeline:
    """
    Full RAG pipeline combining vector store retrieval
    and LLM generation with conversation history.
    """

    def __init__(self):
        from src.retrieval.vector_store import build_vector_store
        from src.generation.llm_engine import LLMEngine

        print("Initializing RAG Pipeline...")
        self.vector_store = build_vector_store()
        self.llm = LLMEngine()
        self.conversation_history = []
        self.log_dir = os.path.join(config.BASE_DIR, "logs", "conversations")
        os.makedirs(self.log_dir, exist_ok=True)
        print("RAG Pipeline ready.")

    def query(self, user_query: str, top_k: int = None) -> dict:
        """
        Process a user query through the full RAG pipeline.
        1. Retrieve relevant documents
        2. Generate answer using LLM
        3. Log conversation
        """
        start = time.time()
        top_k = top_k or config.TOP_K_RESULTS

        # Retrieve relevant documents
        context_docs = self.vector_store.search(user_query, top_k=top_k)

        # Generate answer
        if context_docs:
            result = self.llm.generate(user_query, context_docs)
            retrieval_status = "success"
        else:
            result = self.llm.generate_without_context(user_query)
            retrieval_status = "no_context_found"

        latency_ms = round((time.time() - start) * 1000, 2)

        # Build response
        response = {
            "query": user_query,
            "answer": result["answer"],
            "sources": result["context_used"],
            "num_sources": result["num_context_docs"],
            "retrieval_status": retrieval_status,
            "latency_ms": latency_ms,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_query,
            "timestamp": response["timestamp"],
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["context_used"],
            "timestamp": response["timestamp"],
        })

        return response

    def get_conversation_history(self) -> list:
        """Return the full conversation history."""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")

    def save_conversation(self):
        """Save conversation to disk."""
        if not self.conversation_history:
            print("No conversation to save.")
            return None

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.log_dir, f"conversation_{timestamp}.json")
        with open(path, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "turns": len(self.conversation_history) // 2,
                "history": self.conversation_history,
            }, f, indent=2)
        print(f"Conversation saved to: {path}")
        return path

    def print_conversation(self):
        """Print conversation history in a readable format."""
        print("\n" + "=" * 60)
        print("  CONVERSATION HISTORY")
        print("=" * 60)
        for entry in self.conversation_history:
            if entry["role"] == "user":
                print(f"\nUser: {entry['content']}")
            else:
                print(f"Assistant: {entry['content']}")
                if entry.get("sources"):
                    print(f"Sources: {', '.join(entry['sources'])}")
        print("=" * 60)


if __name__ == "__main__":
    pipeline = RAGPipeline()

    # Simulate a multi-turn conversation
    queries = [
        "What is the total revenue across all customers?",
        "Which region generates the most revenue?",
        "What is the product utilization rate?",
        "How has revenue been trending recently?",
        "How are enterprise customers performing compared to SMB?",
    ]

    print("\n--- Multi-turn RAG Conversation ---")
    for query in queries:
        print(f"\nUser: {query}")
        response = pipeline.query(query)
        print(f"Assistant: {response['answer']}")
        print(f"Sources: {response['sources']}")
        print(f"Latency: {response['latency_ms']}ms")

    pipeline.print_conversation()
    pipeline.save_conversation()