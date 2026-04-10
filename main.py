from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import re
import subprocess

from src.agent import KnowledgeBaseAgent
from src.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    DocumentStructureChunker,
)
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
import json
from src.models import Document
from src.store import EmbeddingStore

# SAMPLE_FILES = [
#     "data/python_intro.txt",
#     "data/vector_store_notes.md",
#     "data/rag_system_design.md",
#     "data/customer_support_playbook.txt",
#     "data/chunking_experiment_report.md",
#     "data/vi_retrieval_notes.md",
# ]

DATA = [
    "data/luat_lao_dong.md",
]

# Change this single line to switch chunking strategy/configuration.
# Pick one of the example `CHUNKER` assignments below and uncomment it.
# Available chunkers (see src/chunking.py for details):
# - FixedSizeChunker(chunk_size: int = 500, overlap: int = 50)
# - SentenceChunker(max_sentences_per_chunk: int = 3)
# - RecursiveChunker(separators: list[str] | None = None, chunk_size: int = 500)
# - DocumentStructureChunker(max_sections_per_chunk: int = 3, fallback_chunk_size: int = 500)
# Examples (uncomment one to activate):
# CHUNKER = SentenceChunker(max_sentences_per_chunk=3)
# CHUNKER = FixedSizeChunker(chunk_size=500, overlap=50)
# CHUNKER = RecursiveChunker(chunk_size=500)
# CHUNKER = RecursiveChunker(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=400)
CHUNKER = DocumentStructureChunker(max_sections_per_chunk=3, fallback_chunk_size=500)
# CHUNKER = SentenceChunker(max_sentences_per_chunk=3)


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing.

    This mock extracts the provided context from the prompt and returns a
    short, human-readable summary instead of echoing the whole prompt.
    """
    m = re.search(r"Use the following context to answer the question:\s*(.*?)\s*Question:", prompt, flags=re.S)
    if m:
        context = re.sub(r"\s+", " ", m.group(1)).strip()
        # Split into rough sentences and return the first two as a mock summary
        sentences = re.split(r"(?<=[.!?])\s+", context)
        summary = " ".join(s.strip() for s in sentences if s)[:600]
        return f"[DEMO LLM] Mock answer (summary): {summary}"
    # Fallback: short preview
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer preview: {preview}..."


def ollama_llm(prompt: str, model: str | None = None) -> str:
    """Call a local Ollama model via the `ollama` CLI.

    Notes:
      - Requires `ollama` to be installed and the model available locally.
      - Configure model with the `OLLAMA_MODEL` env var or pass `model`.
    """
    model = model or os.getenv("OLLAMA_MODEL")
    if not model:
        return "[OLLAMA] No model specified (set OLLAMA_MODEL env var)."

    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if proc.returncode != 0:
            return f"[OLLAMA ERROR] {proc.stderr.strip() or proc.stdout.strip()}"
        return proc.stdout.strip()
    except FileNotFoundError:
        return "[OLLAMA] ollama CLI not found; install Ollama to use this backend."
    except Exception as e:
        return f"[OLLAMA ERROR] {e}"


def chunk_documents(docs: list[Document], chunker) -> list[Document]:
    """Apply the configured chunker to a list of Documents and return chunked Documents.

    Each chunk gets its own id and `chunk_index` metadata, and preserves original `doc_id`.
    """
    if chunker is None:
        return docs

    out: list[Document] = []
    for doc in docs:
        chunks = chunker.chunk(doc.content)
        if not chunks:
            continue
        if len(chunks) == 1:
            out.append(Document(id=doc.id, content=chunks[0], metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": 0}))
        else:
            for i, c in enumerate(chunks):
                out.append(Document(id=f"{doc.id}_{i}", content=c, metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": i}))
    return out


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or DATA

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    # Default to using the local Sentence-Transformers embedder (all-MiniLM-L6-v2)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "local").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception as e:
            print(f"Failed to initialize LocalEmbedder: {e}")
            print("Install 'sentence-transformers' and the model, or set EMBEDDING_PROVIDER to 'openai'.")
            return 1
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception as e:
            print(f"Failed to initialize OpenAI embedder: {e}")
            print("Ensure OpenAI credentials are configured or use the local embedder.")
            return 1
    else:
        # Default to local embedder by name; fail loudly if unavailable
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception as e:
            print(f"Failed to initialize default LocalEmbedder: {e}")
            print("Install 'sentence-transformers' and the model, or set EMBEDDING_PROVIDER explicitly.")
            return 1

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    # LLM selection: set LLM_BACKEND=ollama to use local Ollama model (requires OLLAMA_MODEL env var)
    llm_provider = os.getenv("LLM_BACKEND", "ollama").strip().lower()
    if llm_provider == "ollama":
        llm_fn = lambda p: ollama_llm(p, model=os.getenv("OLLAMA_MODEL"))
    else:
        llm_fn = demo_llm
    print(f"\nLLM backend: {llm_provider}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)

    # Chunk documents using the configured CHUNKER (change CHUNKER near the top of this file)
    chunked_docs = chunk_documents(docs, CHUNKER)
    store.add_documents(chunked_docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")

    # If a question was passed on the command line, run a single query and exit.
    if question:
        query = question
        print("\n=== EmbeddingStore Search Test ===")
        print(f"Query: {query}")
        search_results = store.search(query, top_k=3)
        for index, result in enumerate(search_results, start=1):
            print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
            print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

        print("\n=== KnowledgeBaseAgent Test ===")
        agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
        print(f"Question: {query}")
        print("Agent answer:")
        print(agent.answer(query, top_k=3))
        return 0

    # Otherwise, enter an interactive chatbot-style loop.
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    print("\nEntering interactive chat mode. Type 'exit' or 'quit' to stop.")
    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break

            # No slash-commands: treat all input as user queries.
            # Exit commands like 'exit'/'quit' are still handled above.

            # Treat input as a user query. Append a short instruction to prefer Vietnamese-only output.
            # Append language instruction for the LLM, but use the raw user_input
            # as the retrieval query so the embedding search isn't contaminated
            instruction = "\n\nVui lòng trả lời bằng tiếng Việt; không sử dụng tiếng Trung Quốc."
            agent_query = user_input + instruction
            answer = agent.answer(agent_query, top_k=3, search_query=user_input)

            # Print the agent's answer (LLM output)
            print("\nAgent answer:")
            print(answer.strip())

            # Retrieve latest results
            retrieved = getattr(agent, "_last_retrieved_chunks", None)
            if not retrieved:
                retrieved = store.search(user_input, top_k=3)

            # Pretty ASCII table for top-3 retrievals
            if retrieved:
                rank_w = 4
                score_w = 7
                chunk_w = 100
                sep = "+" + "-" * (rank_w + 2) + "+" + "-" * (score_w + 2) + "+" + "-" * (chunk_w + 2) + "+"
                print(sep)
                print(f"| {'Rank':^{rank_w}} | {'Score':^{score_w}} | {'Top chunk':^{chunk_w}} |")
                print(sep)
                for idx, r in enumerate(retrieved[:3], start=1):
                    score = r.get("score", 0.0) or 0.0
                    content = (r.get("content") or "").replace("\n", " ").strip()
                    preview = content[:chunk_w]
                    print(f"| {idx:^{rank_w}} | {score:{score_w}.3f} | {preview:<{chunk_w}} |")
                print(sep)
    except Exception as e:
        print(f"\nError during interactive loop: {e}")

    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
