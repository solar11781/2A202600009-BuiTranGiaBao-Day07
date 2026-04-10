from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # TODO: split into sentences, group into chunks
        if not text:
            return []

        # Normalize line breaks that follow a sentence-ending period so
        # that ".\n" behaves like ". ". Then split on sentence boundaries
        # defined as punctuation followed by whitespace.
        text = text.strip().replace(".\n", ". ")
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        chunks: list[str] = []
        step = self.max_sentences_per_chunk
        for i in range(0, len(sentences), step):
            chunk = " ".join(sentences[i : i + step]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # TODO: implement recursive splitting strategy
        if not text:
            return []
        return [c for c in self._split(text, list(self.separators)) if c]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # TODO: recursive helper used by RecursiveChunker.chunk
        # Base case: if text is short enough, return as-is
        current_text = current_text.strip()
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # If no separators left or the only separator is empty string,
        # fall back to fixed-size chopping.
        if not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        sep = remaining_separators[0]
        rest = remaining_separators[1:]

        # Empty-string separator is a signal to do fixed-size splitting
        if sep == "":
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        # If separator not present, try next separator
        if sep not in current_text:
            return self._split(current_text, rest)

        parts = current_text.split(sep)
        outputs: list[str] = []
        for idx, part in enumerate(parts):
            # Re-attach the separator except for the last part so that
            # separators (like '. ' or '\n\n') are preserved in the context
            piece = part + (sep if idx < len(parts) - 1 else "")
            piece = piece.strip()
            if not piece:
                continue
            # Recurse on each piece with the remaining separators
            outputs.extend(self._split(piece, rest))

        return outputs


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # TODO: implement cosine similarity formula
    if not vec_a or not vec_b:
        return 0.0

    dot = _dot(vec_a, vec_b)
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # TODO: call each chunker, compute stats, return comparison dict
        fixed = FixedSizeChunker(chunk_size=chunk_size).chunk(text)
        by_sentences = SentenceChunker(max_sentences_per_chunk=3).chunk(text)
        recursive = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def _stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_length = float(sum(len(c) for c in chunks) / count) if count > 0 else 0.0
            return {"count": count, "avg_length": avg_length, "chunks": chunks}

        return {
            "fixed_size": _stats(fixed),
            "by_sentences": _stats(by_sentences),
            "recursive": _stats(recursive),
        }


class DocumentStructureChunker:
    """
    Chunk text by document structure (headings / sections).

    Detection rules (defaults):
      - Markdown headings starting with `#` (levels 1-6).
      - Lines starting with `Section`, `Chapter`, or `Part` (case-insensitive).

    Behavior:
      - Split the document into sections at detected headings.
      - Group up to `max_sections_per_chunk` consecutive sections into a chunk.
      - If no headings are detected, fall back to fixed-size chunking using
        `fallback_chunk_size`.
    """

    def __init__(self, max_sections_per_chunk: int = 3, fallback_chunk_size: int = 500) -> None:
        self.max_sections_per_chunk = max(1, max_sections_per_chunk)
        self.fallback_chunk_size = fallback_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Use line-aware parsing to detect headings/sections
        lines = text.splitlines(keepends=True)
        # Detect Markdown headings (e.g. '#', '##', '###') only.
        md_header_re = re.compile(r"^\s{0,3}(#{1,6})\s+.*$")

        sections: list[str] = []
        current: list[str] = []
        found_md_heading = False

        for line in lines:
            # Only treat explicit Markdown headings as section boundaries.
            if md_header_re.match(line):
                if current:
                    sections.append("".join(current).strip())
                current = [line]
                found_md_heading = True
            else:
                current.append(line)

        if current:
            sections.append("".join(current).strip())

        # If we didn't find any Markdown headings, fall back to fixed-size
        if not found_md_heading or len(sections) == 0:
            return FixedSizeChunker(chunk_size=self.fallback_chunk_size, overlap=0).chunk(text)

        # Group adjacent sections into chunks
        chunks: list[str] = []
        step = self.max_sections_per_chunk
        for i in range(0, len(sections), step):
            chunk = "\n\n".join(s for s in sections[i : i + step] if s).strip()
            if chunk:
                chunks.append(chunk)
        return chunks
