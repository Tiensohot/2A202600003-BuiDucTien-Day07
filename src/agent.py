from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        chunks = self._store.search(question, top_k=top_k)

        if not chunks:
            context = "No relevant context found."
        else:
            context = "\n\n".join(
                f"[{i + 1}] {chunk['content']}"
                for i, chunk in enumerate(chunks)
            )

        prompt = (
            "You are a helpful assistant. "
            "Answer the question using ONLY the context provided below. "
            "If the context does not contain enough information, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        return self._llm_fn(prompt)
        raise NotImplementedError("Implement KnowledgeBaseAgent.answer")
