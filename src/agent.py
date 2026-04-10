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
        # TODO: store references to store and llm_fn
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3, search_query: str | None = None) -> str:
        # Use an explicit search_query for retrieval when provided so callers
        # can augment the LLM-facing question (e.g., append language hints)
        # without contaminating the retrieval embedding.
        retrieval_query = search_query if search_query is not None else question

        # Retrieve top-k chunks (with scores/metadata) and record them for inspection.
        results = self.store.search(retrieval_query, top_k=top_k)
        # Save the retrieved records for later inspection by callers (main.py)
        self._last_retrieved_chunks = results

        # Build a simple context prompt including source and score for each chunk.
        context_parts: list[str] = []
        for r in results:
            src = (r.get("metadata") or {}).get("source") or r.get("id") or "unknown"
            score = r.get("score") or 0.0
            content = r.get("content") or ""
            context_parts.append(f"---\nSource: {src} | Score: {score:.3f}\n{content}")

        prompt_ctx = "\n\n".join(context_parts)

        grounding_instructions = (
            "INSTRUCTIONS: Use ONLY the following CONTEXT to answer the Question. "
            "Do NOT use any external knowledge, internet, or your pretraining. "
            "If the answer is not explicitly present in the CONTEXT or cannot be "
            "directly inferred from it, reply you don't know. "
            "Cite sources present in the context when possible by including the source path.\n\n"
        )

        prompt = (
            grounding_instructions
            + "CONTEXT:\n"
            + prompt_ctx
            + "\n\nQuestion: "
            + question
            + "\n\nAnswer:"
        )

        # Save the actual prompt sent to the LLM for debugging/inspection.
        self._last_prompt = prompt

        # Call the LLM and return its response.
        return self.llm_fn(prompt)