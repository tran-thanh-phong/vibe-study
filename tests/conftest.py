import pytest
from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms.mock import MockLLM
from llama_index.core import Settings


@pytest.fixture(autouse=True)
def mock_llm_and_embeddings():
    Settings.llm = MockLLM()
    # 1536 = text-embedding-3-small dimensions; keeps vector math consistent with prod
    Settings.embed_model = MockEmbedding(embed_dim=1536)
    yield
    # Reset private fields directly — the property setter resolves None to
    # MockEmbedding(embed_dim=1) which would pollute subsequent tests.
    Settings._llm = None
    Settings._embed_model = None


@pytest.fixture
def sample_documents():
    return [
        Document(text="Python is a high-level programming language.", metadata={"source": "pdf", "page": 1}),
        Document(text="Machine learning uses statistical methods.", metadata={"source": "youtube", "url": "https://youtube.com/watch?v=abc123"}),
        Document(text="def hello():\n    return 'world'", metadata={"source": "github", "file": "main.py"}),
    ]
