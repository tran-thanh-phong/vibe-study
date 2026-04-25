from functools import lru_cache

import chromadb
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore

_CHROMA_PATH = "./chroma_db"
_COLLECTION_NAME = "learning_materials"


@lru_cache(maxsize=8)
def _client_for(path: str):
    return chromadb.PersistentClient(path=path)


def _client():
    return _client_for(_CHROMA_PATH)


def _collection():
    return _client().get_or_create_collection(_COLLECTION_NAME)


def _collection_size() -> int:
    return _collection().count()


def load_or_create_index() -> VectorStoreIndex:
    """Return an index wrapping the Chroma collection (empty or populated)."""
    vector_store = ChromaVectorStore(chroma_collection=_collection())
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def insert_documents(index: VectorStoreIndex, docs: list[Document]) -> int:
    """Insert documents into the index. Returns number of nodes inserted."""
    if not docs:
        return 0
    before = _collection_size()
    for d in docs:
        index.insert(d)
    after = _collection_size()
    return after - before


def delete_source(index: VectorStoreIndex, source_id: str) -> None:
    """Delete all chunks for a given source_id via Chroma metadata filter."""
    _collection().delete(where={"source_id": source_id})
