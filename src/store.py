from __future__ import annotations

from typing import Any, Callable

from .chunking import VietnamLawArticleChunker, _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    Documents are automatically chunked before embedding using VietnamLawArticleChunker.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
        chunker=None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._chunker = chunker or VietnamLawArticleChunker(max_chars=3000)
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            client = chromadb.Client()
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass
            self._collection = client.create_collection(collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": doc.metadata or {},
        }

    def _search_records(
        self, query: str, records: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        query_vec = self._embedding_fn(query)
        scored = [
            (compute_similarity(query_vec, r["embedding"]), r)
            for r in records
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {**r, "score": round(score, 4)}
            for score, r in scored[:top_k]
        ]

        raise NotImplementedError("Implement EmbeddingStore._search_records")

    def add_documents(self, docs: list[Document]) -> None:
        """
        Chunk, embed, and store each document.

        Each document is split into chunks before embedding so that large documents
        don't exceed the embedding model's token limit (8192 tokens for OpenAI).

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            chunks = self._chunker.chunk(doc.content)
            if not chunks:
                chunks = [doc.content]

            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_doc = Document(
                    id=f"{doc.id}_chunk{chunk_idx}",
                    content=chunk_text,
                    metadata={**(doc.metadata or {}), "chunk_index": chunk_idx, "doc_id": doc.id},
                )
                record = self._make_record(chunk_doc)
                self._next_index += 1

                if self._use_chroma:
                    self._collection.add(
                        ids=[record["id"]],
                        documents=[record["content"]],
                        embeddings=[record["embedding"]],
                        metadatas=[record["metadata"]],
                    )
                else:
                    self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            query_vec = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=min(top_k, self._collection.count()),
            )
            return [
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": round(1 - results["distances"][0][i], 4),
                }
                for i in range(len(results["ids"][0]))
            ]

        return self._search_records(query, self._store, top_k)

        raise NotImplementedError("Implement EmbeddingStore.search")

    def get_collection_size(self) -> int:
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)
        raise NotImplementedError("Implement EmbeddingStore.get_collection_size")

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma:
            query_vec = self._embedding_fn(query)
            kwargs = dict(
                query_embeddings=[query_vec],
                n_results=min(top_k, self._collection.count()),
            )
            if metadata_filter:
                kwargs["where"] = metadata_filter
            results = self._collection.query(**kwargs)
            return [
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": round(1 - results["distances"][0][i], 4),
                }
                for i in range(len(results["ids"][0]))
            ]

        filtered = self._store
        if metadata_filter:
            filtered = [
                r for r in self._store
                if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        return self._search_records(query, filtered, top_k)
        raise NotImplementedError("Implement EmbeddingStore.search_with_filter")

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            results = self._collection.get(where={"doc_id": doc_id})
            ids = results.get("ids", [])
            if not ids:
                return False
            self._collection.delete(ids=ids)
            return True

        before = len(self._store)
        self._store = [
            r for r in self._store if r["metadata"].get("doc_id") != doc_id
        ]
        return len(self._store) < before
        raise NotImplementedError("Implement EmbeddingStore.delete_document")
