import pytest
from llama_index.core import Document


@pytest.fixture
def chroma_dir(tmp_path, monkeypatch):
    """Point the vector store at an isolated chroma dir per test."""
    monkeypatch.setattr("vector_store._CHROMA_PATH", str(tmp_path / "chroma"))
    return tmp_path / "chroma"


def _doc(text, sid, category="Day 1", page=1):
    return Document(
        text=text,
        metadata={"source": "pdf", "page": page, "source_id": sid, "category": category},
    )


class TestVectorStore:
    def test_load_or_create_empty_index(self, chroma_dir):
        from vector_store import load_or_create_index
        index = load_or_create_index()
        assert index is not None

    def test_insert_documents_returns_chunk_count(self, chroma_dir):
        from vector_store import load_or_create_index, insert_documents
        index = load_or_create_index()
        count = insert_documents(index, [_doc("hello world", "s1"), _doc("foo bar", "s1")])
        assert count == 2

    def test_delete_source_removes_chunks(self, chroma_dir):
        from vector_store import load_or_create_index, insert_documents, delete_source, _collection, _collection_size
        index = load_or_create_index()
        insert_documents(index, [_doc("a", "s1"), _doc("b", "s1"), _doc("c", "s2")])
        assert _collection_size() == 3
        delete_source(index, "s1")
        assert _collection_size() == 1
        # Survivor identity check: the remaining chunk must be from s2,
        # proving the delete targeted by metadata key (not by position / not all).
        remaining = _collection().get(include=["metadatas"])
        assert all(m["source_id"] == "s2" for m in remaining["metadatas"])

    def test_load_rehydrates_existing_collection(self, chroma_dir):
        from vector_store import load_or_create_index, insert_documents, _client_for, _collection_size
        index1 = load_or_create_index()
        insert_documents(index1, [_doc("persisted", "s1")])
        # Simulate a fresh process: drop the cached Chroma client so the
        # second load_or_create_index spins up a brand-new one against the
        # same on-disk path.
        _client_for.cache_clear()
        index2 = load_or_create_index()
        assert _collection_size() == 1
        assert index2 is not None
