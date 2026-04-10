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


import re

class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # Split on sentence-ending punctuation followed by space or newline
        # Use regex to split while keeping the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text.strip())
        
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into chunks
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i:i + self.max_sentences_per_chunk]
            chunk = ' '.join(group).strip()
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
        if not text.strip():
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: text fits within chunk_size
        if len(current_text) <= self.chunk_size:
            return [current_text.strip()] if current_text.strip() else []

        # Base case: no separators left — force-split by character
        if not remaining_separators:
            return [
                current_text[i:i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        sep, *rest_seps = remaining_separators

        # If separator not found, try the next one
        if sep not in current_text:
            return self._split(current_text, rest_seps)

        # Split on current separator
        if sep == "":
            parts = list(current_text)
        else:
            raw_parts = current_text.split(sep)
            # Re-attach separator to all parts except the last
            parts = [p + sep for p in raw_parts[:-1]] + [raw_parts[-1]]

        # Merge small parts greedily, recurse on oversized ones
        chunks = []
        buffer = ""

        for part in parts:
            if not part:
                continue
            candidate = buffer + part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                # Flush buffer
                if buffer.strip():
                    chunks.append(buffer.strip())
                # Part itself is too large — recurse with next separators
                if len(part) > self.chunk_size:
                    chunks.extend(self._split(part, rest_seps))
                    buffer = ""
                else:
                    buffer = part

        if buffer.strip():
            chunks.append(buffer.strip())

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)
    raise NotImplementedError("Implement compute_similarity")


class VietnamLawArticleChunker:
    """Custom chunking strategy for Vietnamese legal documents (Thông tư, Nghị định, etc.).

    Design rationale: Vietnamese legal texts are structured around **Điều** (Articles)
    as the primary semantic unit. Each Điều is a self-contained legal provision with a
    numbered title (e.g., "**Điều 5. Trách nhiệm nhập và đăng tải...**"). Retrieval
    queries in a law domain almost always target a specific article or concept governed
    by one article, so article-aligned chunks preserve the legal context that sentence-
    or character-based chunkers would split mid-provision.

    Algorithm:
        1. Split text on the `**Điều N.` pattern to isolate each article.
        2. If an article body exceeds `max_chars`, fall back to splitting on Khoản
           boundaries (lines starting with a digit followed by a period, e.g. "1. ").
        3. Prepend the article header to every sub-chunk so retrieval results are
           self-contained and traceable.
        4. A preamble block (anything before the first Điều) is kept as one chunk.
    """

    def __init__(self, max_chars: int = 1200) -> None:
        self.max_chars = max_chars
        # Matches "**Điều 5. Title**" or "**Điều 10. Title**" as a line
        self._article_re = re.compile(r"(?m)(?=\*\*\s*[Đđ]i[eề]u\s+\d+[\.\:])")
        # Matches clause start "1. " or "2. " at line beginning
        self._khoan_re = re.compile(r"(?m)(?=^\d+\.\s)", re.MULTILINE)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []

        parts = self._article_re.split(text)
        chunks: list[str] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.max_chars:
                chunks.append(part)
            else:
                # Article is too long — split by Khoản (numbered clauses)
                header_match = re.match(r"(\*\*[^\n]+\*\*)", part)
                header = header_match.group(1) + "\n" if header_match else ""
                sub_parts = self._khoan_re.split(part)
                buffer = ""
                for sub in sub_parts:
                    sub = sub.strip()
                    if not sub:
                        continue
                    candidate = (buffer + "\n" + sub).strip() if buffer else sub
                    if len(candidate) <= self.max_chars:
                        buffer = candidate
                    else:
                        if buffer:
                            chunks.append(header + buffer if header not in buffer else buffer)
                        # Sub-chunk still oversized (e.g., appendix tables) — fixed-size fallback
                        if len(sub) > self.max_chars:
                            for i in range(0, len(sub), self.max_chars):
                                piece = (header + sub[i : i + self.max_chars]).strip()
                                if piece:
                                    chunks.append(piece)
                            buffer = ""
                        else:
                            buffer = sub
                if buffer:
                    chunks.append(header + buffer if header not in buffer else buffer)

        return [c for c in chunks if c.strip()]


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        results = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            lengths = [len(c) for c in chunks]
            results[name] = {
                "chunks": chunks,
                "count": len(chunks),
                "avg_length": sum(lengths) / len(lengths) if lengths else 0.0,
                "min_chunk_size": min(lengths) if lengths else 0,
                "max_chunk_size": max(lengths) if lengths else 0,
            }

        return results
