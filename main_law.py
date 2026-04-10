from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "raw_data/15_2026_TT-BTC_696722.md",
    "raw_data/167_2012_TT-BTC_149305.md",
    "raw_data/19_2014_TT-BTP_249771.md",
    "raw_data/25_2014_TT-BTP_262503.md",
    "raw_data/63_2010_ND-CP_106929.md",
    "raw_data/66_7_2025_NQ-CP_681135.md",
]

SAMPLE_QUESTIONS = [
    "Thông tư 15/2026/TT-BTC hướng dẫn nguyên tắc kế toán cho loại tổ chức nào tham gia thị trường tài sản mã hóa tại Việt Nam?",
    "Theo Nghị định 63/2010/NĐ-CP, 'kiểm soát thủ tục hành chính' được định nghĩa như thế nào?",
    "Nghị quyết 66.7/2025/NQ-CP cho phép thay thế những loại giấy tờ nào bằng thông tin khai thác từ Cơ sở dữ liệu quốc gia về dân cư?",
    "Theo Thông tư 25/2014/TT-BTP, tổ chức được kiểm tra phải gửi báo cáo cho Đoàn kiểm tra trước bao nhiêu ngày làm việc?",
    "Khi nào cơ quan giải quyết thủ tục hành chính được phép yêu cầu cá nhân, tổ chức bổ sung thành phần hồ sơ theo quy định của Nghị quyết 66.7/2025/NQ-CP?",
]


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
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

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
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} chunks in EmbeddingStore")

    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    questions = [query] if question else SAMPLE_QUESTIONS

    print("\n=== KnowledgeBaseAgent Q&A ===")
    for i, q in enumerate(questions, start=1):
        print(f"\n[Câu {i}] {q}")
        search_results = store.search(q, top_k=3)
        print("  Top chunks retrieved:")
        for idx, result in enumerate(search_results, start=1):
            src = result['metadata'].get('source', '')
            print(f"    {idx}. score={result['score']:.3f} | {src}")
            print(f"       {result['content'][:120].replace(chr(10), ' ')}...")
        print("  Agent answer:")
        print(" ", agent.answer(q, top_k=3))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
