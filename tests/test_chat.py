import pytest
from llama_index.core.chat_engine.types import BaseChatEngine


class TestCreateChatEngine:
    def test_returns_chat_engine(self, sample_documents):
        from vector_store import build_index
        from chat import create_chat_engine
        index = build_index(sample_documents)
        engine = create_chat_engine(index)
        assert isinstance(engine, BaseChatEngine)


class TestChat:
    def test_returns_string_response(self, sample_documents):
        from vector_store import build_index
        from chat import create_chat_engine, chat
        index = build_index(sample_documents)
        engine = create_chat_engine(index)
        response = chat(engine, "What is Python?")
        assert isinstance(response, str)
        assert len(response) > 0
