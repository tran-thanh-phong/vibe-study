# AI Learning Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit app that ingests PDF/YouTube/GitHub content and provides RAG chat, outline generation, and exercise generation powered by LlamaIndex + gpt-4o-mini.

**Architecture:** Modular Python backend (one file per concern) with a Streamlit frontend that manages all state via `st.session_state`. LlamaIndex `VectorStoreIndex` (backed by local Chroma) handles RAG chat; `SummaryIndex` handles full-corpus outline and exercise generation.

**Tech Stack:** Python 3.11+, LlamaIndex, Chroma, OpenAI (gpt-4o-mini + text-embedding-3-small), Streamlit, pypdf, youtube-transcript-api, requests

---

## File Map

| File | Responsibility |
|------|---------------|
| `requirements.txt` | All Python dependencies |
| `.env.example` | Template for OPENAI_API_KEY |
| `ingestion.py` | PDF / YouTube / GitHub → `list[Document]` |
| `vector_store.py` | Build Chroma-backed `VectorStoreIndex` from documents |
| `chat.py` | Create `CondensePlusContextChatEngine`; wrap `.chat()` |
| `outline.py` | Generate structured outline via `SummaryIndex` |
| `exercises.py` | Generate easy/medium/hard exercises via `SummaryIndex` |
| `app.py` | Streamlit UI — sidebar inputs, 3 tabs, session state |
| `tests/conftest.py` | Shared fixtures (mock documents, mock embeddings) |
| `tests/test_ingestion.py` | Unit tests for PDF/YouTube/GitHub parsers |
| `tests/test_vector_store.py` | Unit test for index building |
| `tests/test_chat.py` | Unit tests for chat engine creation and response |
| `tests/test_outline.py` | Unit test for outline generation |
| `tests/test_exercises.py` | Unit test for exercise generation |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `requirements.txt`**

```
llama-index==0.12.8
llama-index-vector-stores-chroma==0.4.1
llama-index-embeddings-openai==0.3.1
llama-index-llms-openai==0.3.3
chromadb==0.6.3
pypdf==5.4.0
youtube-transcript-api==1.0.3
streamlit==1.45.0
python-dotenv==1.1.0
requests==2.32.3
pytest==8.3.5
```

> Note: If exact versions conflict during install, remove version pins and let pip resolve. Re-pin after successful install with `pip freeze`.

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install without errors.

- [ ] **Step 3: Create `.gitignore`**

```
.env
chroma_db/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Create `.env.example`**

```
OPENAI_API_KEY=your-api-key-here
```

- [ ] **Step 5: Copy to `.env` and fill in your real key**

Run: `cp .env.example .env`
Then edit `.env` and set `OPENAI_API_KEY=sk-...`.

- [ ] **Step 5: Create `tests/__init__.py`**

Empty file — makes `tests/` a Python package.

```python
```

- [ ] **Step 6: Create `tests/conftest.py`**

```python
import pytest
from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms.mock import MockLLM
from llama_index.core import Settings


@pytest.fixture(autouse=True)
def mock_llm_and_embeddings():
    Settings.llm = MockLLM()
    Settings.embed_model = MockEmbedding(embed_dim=1536)
    yield
    Settings.llm = None
    Settings.embed_model = None


@pytest.fixture
def sample_documents():
    return [
        Document(text="Python is a high-level programming language.", metadata={"source": "pdf", "page": 1}),
        Document(text="Machine learning uses statistical methods.", metadata={"source": "youtube", "url": "https://youtube.com/watch?v=abc123"}),
        Document(text="def hello():\n    return 'world'", metadata={"source": "github", "file": "main.py"}),
    ]
```

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore .env.example tests/__init__.py tests/conftest.py
git commit -m "chore: project setup with dependencies and test fixtures"
```

---

## Task 2: PDF Ingestion

**Files:**
- Create: `ingestion.py`
- Create: `tests/test_ingestion.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion.py`:

```python
import io
import pytest
from unittest.mock import patch, MagicMock
from llama_index.core import Document


def make_mock_pdf_reader(pages_text):
    mock_reader = MagicMock()
    mock_pages = []
    for text in pages_text:
        page = MagicMock()
        page.extract_text.return_value = text
        mock_pages.append(page)
    mock_reader.pages = mock_pages
    return mock_reader


class TestLoadPdf:
    def test_returns_one_document_per_page(self):
        from ingestion import load_pdf
        mock_reader = make_mock_pdf_reader(["Page one content", "Page two content"])
        with patch("ingestion.PdfReader", return_value=mock_reader):
            docs = load_pdf(b"fake-pdf-bytes")
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].text == "Page one content"
        assert docs[1].text == "Page two content"

    def test_skips_empty_pages(self):
        from ingestion import load_pdf
        mock_reader = make_mock_pdf_reader(["Real content", "   ", ""])
        with patch("ingestion.PdfReader", return_value=mock_reader):
            docs = load_pdf(b"fake-pdf-bytes")
        assert len(docs) == 1
        assert docs[0].text == "Real content"

    def test_metadata_includes_source_and_page(self):
        from ingestion import load_pdf
        mock_reader = make_mock_pdf_reader(["Content"])
        with patch("ingestion.PdfReader", return_value=mock_reader):
            docs = load_pdf(b"fake-pdf-bytes")
        assert docs[0].metadata["source"] == "pdf"
        assert docs[0].metadata["page"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingestion.py::TestLoadPdf -v`
Expected: `ModuleNotFoundError: No module named 'ingestion'`

- [ ] **Step 3: Create `ingestion.py` with `load_pdf()`**

```python
import io
import re
import requests
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi
from llama_index.core import Document


def load_pdf(file_bytes: bytes) -> list[Document]:
    reader = PdfReader(io.BytesIO(file_bytes))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(text=text, metadata={"source": "pdf", "page": i + 1}))
    return docs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingestion.py::TestLoadPdf -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ingestion.py tests/test_ingestion.py
git commit -m "feat: add PDF ingestion"
```

---

## Task 3: YouTube Ingestion

**Files:**
- Modify: `ingestion.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ingestion.py`:

```python
class TestLoadYoutube:
    def test_parses_standard_url(self):
        from ingestion import _parse_video_id
        assert _parse_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_parses_short_url(self):
        from ingestion import _parse_video_id
        assert _parse_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_raises_on_invalid_url(self):
        from ingestion import _parse_video_id
        with pytest.raises(ValueError, match="Could not parse video ID"):
            _parse_video_id("https://example.com/video")

    def test_returns_single_document_with_full_transcript(self):
        from ingestion import load_youtube
        fake_transcript = [
            {"text": "Hello world", "start": 0.0, "duration": 1.0},
            {"text": "This is a test", "start": 1.0, "duration": 1.0},
        ]
        with patch("ingestion.YouTubeTranscriptApi.get_transcript", return_value=fake_transcript):
            docs = load_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert len(docs) == 1
        assert "Hello world" in docs[0].text
        assert "This is a test" in docs[0].text
        assert docs[0].metadata["source"] == "youtube"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingestion.py::TestLoadYoutube -v`
Expected: `ImportError` — `_parse_video_id` and `load_youtube` don't exist yet.

- [ ] **Step 3: Add `_parse_video_id()` and `load_youtube()` to `ingestion.py`**

Add after `load_pdf()`:

```python
def _parse_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Could not parse video ID from URL: {url}")
    return match.group(1)


def load_youtube(url: str) -> list[Document]:
    video_id = _parse_video_id(url)
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join(entry["text"] for entry in transcript_list)
    return [Document(text=text, metadata={"source": "youtube", "url": url})]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingestion.py::TestLoadYoutube -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ingestion.py tests/test_ingestion.py
git commit -m "feat: add YouTube transcript ingestion"
```

---

## Task 4: GitHub Ingestion

**Files:**
- Modify: `ingestion.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ingestion.py`:

```python
class TestLoadGithub:
    def test_parses_github_url(self):
        from ingestion import _parse_github_owner_repo
        owner, repo = _parse_github_owner_repo("https://github.com/openai/openai-python")
        assert owner == "openai"
        assert repo == "openai-python"

    def test_raises_on_invalid_github_url(self):
        from ingestion import _parse_github_owner_repo
        with pytest.raises(ValueError, match="Could not parse owner/repo"):
            _parse_github_owner_repo("https://gitlab.com/user/repo")

    def test_fetches_and_returns_documents(self):
        from ingestion import load_github

        tree_response = MagicMock()
        tree_response.status_code = 200
        tree_response.raise_for_status = MagicMock()
        tree_response.json.return_value = {
            "tree": [
                {"type": "blob", "path": "README.md"},
                {"type": "blob", "path": "main.py"},
                {"type": "blob", "path": "image.png"},  # should be excluded
                {"type": "tree", "path": "src"},         # should be excluded
            ]
        }

        file_response = MagicMock()
        file_response.status_code = 200
        file_response.text = "file content"

        def mock_get(url, **kwargs):
            if "git/trees" in url:
                return tree_response
            return file_response

        with patch("ingestion.requests.get", side_effect=mock_get):
            docs = load_github("https://github.com/owner/repo")

        assert len(docs) == 2
        paths = [d.metadata["file"] for d in docs]
        assert "README.md" in paths
        assert "main.py" in paths

    def test_skips_files_with_empty_content(self):
        from ingestion import load_github

        tree_response = MagicMock()
        tree_response.status_code = 200
        tree_response.raise_for_status = MagicMock()
        tree_response.json.return_value = {"tree": [{"type": "blob", "path": "README.md"}]}

        empty_response = MagicMock()
        empty_response.status_code = 200
        empty_response.text = "   "

        with patch("ingestion.requests.get", side_effect=lambda url, **kw: tree_response if "git/trees" in url else empty_response):
            docs = load_github("https://github.com/owner/repo")

        assert len(docs) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingestion.py::TestLoadGithub -v`
Expected: `ImportError` — `_parse_github_owner_repo` and `load_github` don't exist yet.

- [ ] **Step 3: Add `_parse_github_owner_repo()` and `load_github()` to `ingestion.py`**

Add after `load_youtube()`:

```python
def _parse_github_owner_repo(url: str) -> tuple[str, str]:
    match = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if not match:
        raise ValueError(f"Could not parse owner/repo from URL: {url}")
    return match.group(1), match.group(2)


_ALLOWED_EXTENSIONS = {".py", ".js", ".ts", ".md", ".java", ".go"}
_MAX_GITHUB_FILES = 30


def load_github(url: str) -> list[Document]:
    owner, repo = _parse_github_owner_repo(url)
    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    resp = requests.get(tree_url, headers={"Accept": "application/vnd.github.v3+json"})
    resp.raise_for_status()
    tree = resp.json().get("tree", [])

    docs = []
    for item in tree:
        if len(docs) >= _MAX_GITHUB_FILES:
            break
        if item["type"] != "blob":
            continue
        path = item["path"]
        ext = f".{path.rsplit('.', 1)[-1]}" if "." in path else ""
        if path != "README.md" and ext not in _ALLOWED_EXTENSIONS:
            continue
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
        content_resp = requests.get(raw_url)
        if content_resp.status_code == 200 and content_resp.text.strip():
            docs.append(Document(text=content_resp.text, metadata={"source": "github", "file": path}))
    return docs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingestion.py::TestLoadGithub -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ingestion.py tests/test_ingestion.py
git commit -m "feat: add GitHub repository ingestion"
```

---

## Task 5: `ingest()` Entry Point

**Files:**
- Modify: `ingestion.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ingestion.py`:

```python
class TestIngest:
    def test_combines_all_sources(self):
        from ingestion import ingest
        pdf_doc = Document(text="pdf content", metadata={"source": "pdf", "page": 1})
        yt_doc = Document(text="youtube content", metadata={"source": "youtube", "url": "u"})
        gh_doc = Document(text="github content", metadata={"source": "github", "file": "f"})

        with patch("ingestion.load_pdf", return_value=[pdf_doc]) as mock_pdf, \
             patch("ingestion.load_youtube", return_value=[yt_doc]) as mock_yt, \
             patch("ingestion.load_github", return_value=[gh_doc]) as mock_gh:
            docs = ingest(pdf_bytes=b"x", youtube_url="url", github_url="ghurl")

        assert len(docs) == 3
        mock_pdf.assert_called_once_with(b"x")
        mock_yt.assert_called_once_with("url")
        mock_gh.assert_called_once_with("ghurl")

    def test_skips_none_sources(self):
        from ingestion import ingest
        pdf_doc = Document(text="pdf content", metadata={"source": "pdf", "page": 1})
        with patch("ingestion.load_pdf", return_value=[pdf_doc]) as mock_pdf, \
             patch("ingestion.load_youtube") as mock_yt, \
             patch("ingestion.load_github") as mock_gh:
            docs = ingest(pdf_bytes=b"x")
        assert len(docs) == 1
        mock_yt.assert_not_called()
        mock_gh.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingestion.py::TestIngest -v`
Expected: `ImportError` — `ingest` doesn't exist yet.

- [ ] **Step 3: Add `ingest()` to `ingestion.py`**

Append to `ingestion.py`:

```python
def ingest(
    pdf_bytes: bytes | None = None,
    youtube_url: str | None = None,
    github_url: str | None = None,
) -> list[Document]:
    docs = []
    if pdf_bytes:
        docs.extend(load_pdf(pdf_bytes))
    if youtube_url:
        docs.extend(load_youtube(youtube_url))
    if github_url:
        docs.extend(load_github(github_url))
    return docs
```

- [ ] **Step 4: Run all ingestion tests**

Run: `pytest tests/test_ingestion.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ingestion.py tests/test_ingestion.py
git commit -m "feat: add ingest() entry point combining all sources"
```

---

## Task 6: Vector Store

**Files:**
- Create: `vector_store.py`
- Create: `tests/test_vector_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_vector_store.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vector_store.py -v`
Expected: `ModuleNotFoundError: No module named 'vector_store'`

- [ ] **Step 3: Create `vector_store.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vector_store.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add vector_store.py tests/test_vector_store.py
git commit -m "feat: add Chroma-backed vector store index builder"
```

---

## Task 7: Chat Engine

**Files:**
- Create: `chat.py`
- Create: `tests/test_chat.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chat.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chat.py -v`
Expected: `ModuleNotFoundError: No module named 'chat'`

- [ ] **Step 3: Create `chat.py`**

```python
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer


def create_chat_engine(index) -> CondensePlusContextChatEngine:
    retriever = index.as_retriever(similarity_top_k=5)
    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
    return CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        verbose=False,
    )


def chat(engine: CondensePlusContextChatEngine, user_message: str) -> str:
    response = engine.chat(user_message)
    return str(response)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chat.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add chat.py tests/test_chat.py
git commit -m "feat: add RAG chat engine with conversation history"
```

---

## Task 8: Outline Generation

**Files:**
- Create: `outline.py`
- Create: `tests/test_outline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_outline.py`:

```python
class TestGenerateOutline:
    def test_returns_non_empty_string(self, sample_documents):
        from outline import generate_outline
        result = generate_outline(sample_documents)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_outline.py -v`
Expected: `ModuleNotFoundError: No module named 'outline'`

- [ ] **Step 3: Create `outline.py`**

```python
from llama_index.core import SummaryIndex

_OUTLINE_PROMPT = (
    "Generate a structured outline from the content provided.\n"
    "Format your response as numbered sections:\n"
    "1. [Section Title]\n"
    "   Description: [1-2 sentence description]\n\n"
    "2. [Section Title]\n"
    "   Description: [1-2 sentence description]\n\n"
    "Cover all major topics in the content."
)


def generate_outline(documents) -> str:
    index = SummaryIndex.from_documents(documents)
    query_engine = index.as_query_engine(response_mode="tree_summarize")
    response = query_engine.query(_OUTLINE_PROMPT)
    return str(response)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_outline.py -v`
Expected: 1 test PASS.

- [ ] **Step 5: Commit**

```bash
git add outline.py tests/test_outline.py
git commit -m "feat: add outline generation via SummaryIndex"
```

---

## Task 9: Exercise Generation

**Files:**
- Create: `exercises.py`
- Create: `tests/test_exercises.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_exercises.py`:

```python
class TestGenerateExercises:
    def test_returns_non_empty_string(self, sample_documents):
        from exercises import generate_exercises
        result = generate_exercises(sample_documents)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_exercises.py -v`
Expected: `ModuleNotFoundError: No module named 'exercises'`

- [ ] **Step 3: Create `exercises.py`**

```python
from llama_index.core import SummaryIndex

_EXERCISE_PROMPT = (
    "Generate 3 practice exercises based on the content provided.\n\n"
    "**Easy (Recall/Definition)**\n"
    "Exercise: [state the exercise]\n"
    "Sample Answer: [provide a sample answer]\n\n"
    "**Medium (Application)**\n"
    "Exercise: [state the exercise]\n"
    "Sample Answer: [provide a sample answer]\n\n"
    "**Hard (Analysis/Synthesis)**\n"
    "Exercise: [state the exercise]\n"
    "Sample Answer: [provide a sample answer]"
)


def generate_exercises(documents) -> str:
    index = SummaryIndex.from_documents(documents)
    query_engine = index.as_query_engine(response_mode="tree_summarize")
    response = query_engine.query(_EXERCISE_PROMPT)
    return str(response)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_exercises.py -v`
Expected: 1 test PASS.

- [ ] **Step 5: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add exercises.py tests/test_exercises.py
git commit -m "feat: add exercise generation via SummaryIndex"
```

---

## Task 10: Streamlit UI

**Files:**
- Create: `app.py`

No unit tests for Streamlit UI — verify by running the app manually.

- [ ] **Step 1: Create `app.py`**

```python
import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

from ingestion import ingest
from vector_store import build_index
from chat import create_chat_engine, chat
from outline import generate_outline
from exercises import generate_exercises

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

st.set_page_config(page_title="AI Learning Assistant", layout="wide")
st.title("AI Learning Assistant")

# Initialize session state keys
_STATE_DEFAULTS = {
    "index": None,
    "documents": None,
    "chat_engine": None,
    "messages": [],
    "outline": None,
    "exercises": None,
    "ready": False,
}
for key, default in _STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar: input collection
with st.sidebar:
    st.header("Upload Learning Materials")
    pdf_file = st.file_uploader("PDF Slides", type=["pdf"])
    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    github_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/owner/repo")

    has_input = bool(pdf_file or youtube_url.strip() or github_url.strip())

    if st.button("Process Content", disabled=not has_input):
        with st.spinner("Ingesting and indexing content…"):
            try:
                docs = ingest(
                    pdf_bytes=pdf_file.read() if pdf_file else None,
                    youtube_url=youtube_url.strip() or None,
                    github_url=github_url.strip() or None,
                )
                if not docs:
                    st.error("No content could be extracted from the provided sources.")
                else:
                    st.session_state.documents = docs
                    st.session_state.index = build_index(docs)
                    st.session_state.chat_engine = None
                    st.session_state.messages = []
                    st.session_state.outline = None
                    st.session_state.exercises = None
                    st.session_state.ready = True
            except Exception as e:
                st.error(f"Processing failed: {e}")

    if st.session_state.ready:
        st.success("Ready ✓")

# Main area
if not st.session_state.ready:
    st.info("Upload at least one source in the sidebar, then click **Process Content**.")
    st.stop()

tab_chat, tab_outline, tab_exercises = st.tabs(["💬 Chat", "📋 Outline", "🏋️ Exercises"])

with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.button("Clear chat", key="clear_chat"):
        st.session_state.chat_engine = None
        st.session_state.messages = []
        st.rerun()

    if user_input := st.chat_input("Ask a question about your materials…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if st.session_state.chat_engine is None:
            st.session_state.chat_engine = create_chat_engine(st.session_state.index)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = chat(st.session_state.chat_engine, user_input)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

with tab_outline:
    if st.button("Generate Outline", key="gen_outline"):
        with st.spinner("Generating outline…"):
            try:
                st.session_state.outline = generate_outline(st.session_state.documents)
            except Exception as e:
                st.error(f"Outline generation failed: {e}")
    if st.session_state.outline:
        st.markdown(st.session_state.outline)

with tab_exercises:
    if st.button("Generate Exercises", key="gen_exercises"):
        with st.spinner("Generating exercises…"):
            try:
                st.session_state.exercises = generate_exercises(st.session_state.documents)
            except Exception as e:
                st.error(f"Exercise generation failed: {e}")
    if st.session_state.exercises:
        st.markdown(st.session_state.exercises)
```

- [ ] **Step 2: Run the app**

Run: `streamlit run app.py`
Expected: Browser opens at `http://localhost:8501`.

- [ ] **Step 3: Manual smoke test — PDF only**

1. Upload any PDF in the sidebar
2. Click "Process Content" — spinner should appear, then "Ready ✓"
3. Switch to 💬 Chat tab
4. Ask: "What is this document about?" — expect a relevant answer
5. Ask a follow-up: "Can you elaborate on the first point?" — expect the answer to reference prior context (not start fresh)
6. Click "Clear chat" — chat history should reset
7. Switch to 📋 Outline — click "Generate Outline" — expect a numbered list of sections
8. Switch to 🏋️ Exercises — click "Generate Exercises" — expect 3 labeled exercises

- [ ] **Step 4: Manual smoke test — all three sources**

1. Provide a PDF, a YouTube URL (with captions), and a public GitHub repo URL
2. Click "Process Content"
3. Ask a question that spans sources (e.g., "How does the code in the repo relate to the concepts in the slides?")
4. Verify outline and exercises reflect content from all three sources

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit UI with sidebar, chat tab, outline tab, and exercises tab"
```

---

## Task 11: Final Checks

- [ ] **Step 1: Run the full test suite one final time**

Run: `pytest -v`
Expected: All tests PASS.

- [ ] **Step 2: Commit if any loose files remain**

Run: `git status`
Commit any remaining untracked files with appropriate messages.
