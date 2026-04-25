import pytest
from llama_index.core import Document
from llama_index.core.chat_engine.types import BaseChatEngine


@pytest.fixture
def chroma_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("vector_store._CHROMA_PATH", str(tmp_path / "chroma"))
    return tmp_path / "chroma"


def _tagged_sample_docs():
    return [
        Document(text="Python is a high-level programming language.",
                 metadata={"source": "pdf", "page": 1,
                           "source_id": "s-py", "category": "Day 1"}),
        Document(text="Machine learning uses statistical methods.",
                 metadata={"source": "youtube", "url": "https://youtube.com/watch?v=abc123",
                           "source_id": "s-ml", "category": "Day 2"}),
    ]


class TestCreateChatEngine:
    def test_returns_chat_engine(self, chroma_dir):
        from vector_store import load_or_create_index, insert_documents
        from chat import create_chat_engine
        index = load_or_create_index()
        insert_documents(index, _tagged_sample_docs())
        engine = create_chat_engine(index)
        assert isinstance(engine, BaseChatEngine)


class TestChat:
    def test_returns_string_response(self, chroma_dir):
        from vector_store import load_or_create_index, insert_documents
        from chat import create_chat_engine, chat
        index = load_or_create_index()
        insert_documents(index, _tagged_sample_docs())
        engine = create_chat_engine(index)
        response = chat(engine, "What is Python?")
        assert isinstance(response, str)
        assert len(response) > 0
