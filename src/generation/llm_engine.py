import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from configs.config import config


class LLMEngine:
    """
    LLM generation engine using Google Flan-T5.
    Takes retrieved context and user query,
    generates a natural language answer.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading LLM: {config.LLM_MODEL}")
        print(f"Device: {self.device}")

        self.tokenizer = T5Tokenizer.from_pretrained(config.LLM_MODEL)
        self.model = T5ForConditionalGeneration.from_pretrained(config.LLM_MODEL)
        self.model.to(self.device)
        self.model.eval()
        print("LLM loaded successfully.")

    def build_prompt(self, query: str, context_docs: list) -> str:
        """
        Build a prompt from the query and retrieved context documents.
        """
        context = "\n\n".join([
            f"Document: {doc['title']}\n{doc['content']}"
            for doc in context_docs
        ])

        prompt = f"""Answer the following business analytics question based on the provided context.
Be specific and include numbers where available.

Context:
{context}

Question: {query}

Answer:"""

        return prompt

    def generate(self, query: str, context_docs: list) -> dict:
        """
        Generate an answer given a query and retrieved context.
        """
        prompt = self.build_prompt(query, context_docs)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=config.MAX_NEW_TOKENS,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "query": query,
            "answer": answer,
            "context_used": [doc["title"] for doc in context_docs],
            "num_context_docs": len(context_docs),
        }

    def generate_without_context(self, query: str) -> dict:
        """
        Generate an answer without RAG context.
        Used as fallback when no relevant documents are found.
        """
        prompt = f"Answer this business analytics question: {query}"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "query": query,
            "answer": answer,
            "context_used": [],
            "num_context_docs": 0,
        }


if __name__ == "__main__":
    from src.retrieval.vector_store import build_vector_store

    # Build vector store
    store = build_vector_store()

    # Load LLM
    llm = LLMEngine()

    # Test queries
    test_queries = [
        "What is the total revenue?",
        "Which region is performing the best?",
        "What is the average product utilization rate?",
    ]

    print("\n--- RAG Question Answering ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        context_docs = store.search(query, top_k=3)
        result = llm.generate(query, context_docs)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['context_used']}")