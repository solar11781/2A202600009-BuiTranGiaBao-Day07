from __future__ import annotations

from typing import Any, Callable
import os
import math

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        # Only enable ChromaDB if explicitly requested via env var. This
        # keeps behavior deterministic for tests and classroom runs where
        # a system-wide chroma instance might already contain data.
        use_chroma_env = os.getenv("USE_CHROMA") or os.getenv("CHROMA_ENABLED")
        if use_chroma_env and use_chroma_env.lower() in ("1", "true", "yes"):
            try:
                import chromadb  # noqa: F401
                try:
                    # Newer chromadb may require Settings
                    from chromadb.config import Settings  # type: ignore
                    client = chromadb.Client(Settings())
                except Exception:
                    client = chromadb.Client()

                try:
                    # Create collection with given name (may return existing collection)
                    collection = client.get_collection(self._collection_name)
                except Exception:
                    collection = client.create_collection(self._collection_name)

                self._collection = collection
                # TODO: initialize chromadb client + collection
                self._use_chroma = True
            except Exception:
                self._use_chroma = False
                self._collection = None
        else:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        embedding = self._embedding_fn(doc.content)
        # normalize embedding to unit length when possible so scores are comparable
        try:
            emb_list = [float(x) for x in embedding]
        except Exception:
            emb_list = []
        norm = math.sqrt(sum(x * x for x in emb_list)) if emb_list else 0.0
        if norm and norm != 0.0:
            emb_list = [x / norm for x in emb_list]
        metadata = dict(doc.metadata) if isinstance(doc.metadata, dict) else {}
        metadata.setdefault("doc_id", doc.id)
        record = {
            "internal_id": str(self._next_index),
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": emb_list,
        }
        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        if not records:
            return []
        q_emb = self._embedding_fn(query)
        try:
            q_emb = [float(x) for x in q_emb]
        except Exception:
            q_emb = []
        qnorm = math.sqrt(sum(x * x for x in q_emb)) if q_emb else 0.0
        if qnorm and qnorm != 0.0:
            q_emb = [x / qnorm for x in q_emb]
        results: list[dict[str, Any]] = []
        for r in records:
            emb = r.get("embedding")
            if not emb:
                continue
            # ensure comparable lengths
            try:
                if not q_emb:
                    # no valid query embedding: skip
                    continue
                if len(q_emb) != len(emb):
                    # skip mismatched-dimension records
                    continue
                score = _dot(q_emb, emb)
            except Exception:
                continue
            results.append({
                "id": r.get("id"),
                "content": r.get("content"),
                "metadata": r.get("metadata", {}),
                "score": score,
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        if self._use_chroma and self._collection is not None:
            try:
                ids = [d.id for d in docs]
                documents = [d.content for d in docs]
                # normalize embeddings before inserting into Chroma for consistency
                embeddings = []
                for d in docs:
                    emb = self._embedding_fn(d.content)
                    try:
                        emb_list = [float(x) for x in emb]
                    except Exception:
                        emb_list = []
                    norm = math.sqrt(sum(x * x for x in emb_list)) if emb_list else 0.0
                    if norm and norm != 0.0:
                        emb_list = [x / norm for x in emb_list]
                    embeddings.append(emb_list)

                self._collection.add(ids=ids, documents=documents, embeddings=embeddings)
                # also maintain an in-memory copy for deterministic filtering/search
                for d in docs:
                    rec = self._make_record(d)
                    self._store.append(rec)
                self._next_index += len(docs)
                return
            except Exception:
                # If chroma integration fails, fall back to in-memory
                pass

        for d in docs:
            rec = self._make_record(d)
            self._store.append(rec)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        # If we have a chroma collection try to use it; otherwise use in-memory
        if self._use_chroma and self._collection is not None:
            try:
                q_emb = self._embedding_fn(query)
                resp = self._collection.query(query_embeddings=[q_emb], n_results=top_k)
                # Attempt to normalize response into expected format if possible
                if isinstance(resp, dict):
                    documents = resp.get("documents") or []
                    ids = resp.get("ids") or []
                    distances = resp.get("distances") or []
                    results: list[dict[str, Any]] = []
                    for i, doc_text in enumerate(documents[:top_k]):
                        score = None
                        try:
                            score = distances[0][i]
                        except Exception:
                            score = None
                        results.append({"id": ids[0][i] if ids else None, "content": doc_text, "metadata": {}, "score": score})
                    return results
            except Exception:
                # fallback to in-memory search
                pass

        return self._search_records(query, self._store, top_k)

    # Backwards-compatible alias expected by other modules
    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        results = self.search(query, top_k=top_k)
        # Backwards-compatible: return list of chunk strings for simple agents
        return [r.get("content") for r in results]

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        if self._use_chroma and self._collection is not None:
            try:
                data = self._collection.get()
                ids = data.get("ids") if isinstance(data, dict) else None
                if ids:
                    # ids is often a list-of-lists for collections
                    if isinstance(ids, list) and ids and isinstance(ids[0], list):
                        return len(ids[0])
                    return len(ids)
            except Exception:
                pass
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        if metadata_filter is None:
            return self.search(query, top_k=top_k)

        filtered: list[dict[str, Any]] = []
        for r in self._store:
            md = r.get("metadata", {}) or {}
            match = True
            for k, v in (metadata_filter or {}).items():
                if md.get(k) != v:
                    match = False
                    break
            if match:
                filtered.append(r)

        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        original_len = len(self._store)
        self._store = [r for r in self._store if r.get("metadata", {}).get("doc_id") != doc_id]
        return len(self._store) < original_len
