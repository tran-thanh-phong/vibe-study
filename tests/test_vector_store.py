import pytest
from llama_index.core import Document, VectorStoreIndex


class TestBuildIndex:
    def test_returns_vector_store_index(self, sample_documents):
        from vector_store import build_index
        index = build_index(sample_documents)
        assert isinstance(index, VectorStoreIndex)

    def test_recreates_collection_on_each_call(self, sample_documents):
        from vector_store import build_index
        # Two calls must not raise "collection already exists" errors
        index1 = build_index(sample_documents)
        index2 = build_index(sample_documents)
        assert isinstance(index1, VectorStoreIndex)
        assert isinstance(index2, VectorStoreIndex)
