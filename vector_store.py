import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

_CHROMA_PATH = "./chroma_db"
_COLLECTION_NAME = "learning_materials"


def build_index(documents) -> VectorStoreIndex:
    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)
