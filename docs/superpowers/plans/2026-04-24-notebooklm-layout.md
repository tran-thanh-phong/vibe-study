# NotebookLM-Style Source Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current single-shot sidebar ingestion with an incremental, categorised source library (left pane) + Chat / Outline / Exercises with scope selector (right pane). Library persists across runs.

**Architecture:** Sources are registered in `storage/sources.json` with `source_id` + `category`. Each source's `Document` list is pickled to `storage/docstore/<source_id>.pkl` (needed for outline/exercises). Chunks in Chroma carry `source_id` + `category` metadata so retrieval can filter by scope and deletion can wipe a source atomically. UI is a `st.columns([1, 3])` split: library on the left, tabs on the right.

**Tech Stack:** Python 3.11+, Streamlit 1.45, LlamaIndex 0.12, ChromaDB 0.6.3, pytest 8.3.

**Spec:** `docs/superpowers/specs/2026-04-24-notebooklm-layout-design.md`

---

## File Structure

**New files:**
- `library.py` &mdash; source registry, doc persistence, orchestration (ingest + register, remove)
- `ui/__init__.py` &mdash; empty; marks `ui` as a package
- `ui/source_library.py` &mdash; left-pane rendering + Add Source form
- `ui/workspace.py` &mdash; right-pane rendering (scope selector + tabs)
- `tests/test_library.py`
- `tests/test_vector_store.py`
- `tests/test_chat_scope.py`

**Modified files:**
- `vector_store.py` &mdash; rewritten for persistent, incremental index (`load_or_create_index`, `insert_documents`, `delete_source`)
- `chat.py` &mdash; `create_chat_engine(index, scope)` adds metadata filter when scope is a category
- `ingestion.py` &mdash; unchanged loaders; all tagging happens in `library.py` so existing tests stay green
- `outline.py` &mdash; no change (already takes a `list[Document]`)
- `exercises.py` &mdash; no change (already takes a `list[Document]`)
- `app.py` &mdash; rewritten to the two-column layout; initialises registry + index on startup
- `.gitignore` &mdash; already includes `.superpowers/`; add `storage/` in Task 1

**Directory on disk at runtime:**
```
chroma_db/              # already exists, now preserved between runs
storage/
  ├── sources.json      # registry
  └── docstore/*.pkl    # one pickle per source
```

---

## Task 1: Source dataclass + JSON registry (read / write / mutate)

**Files:**
- Create: `library.py`
- Create: `tests/test_library.py`
- Modify: `.gitignore` (add `storage/`)

- [ ] **Step 1: Add `storage/` to `.gitignore`**

Edit `.gitignore` and append `storage/` on a new line so the registry + pickles are never committed.

- [ ] **Step 2: Write the failing test for registry round-trip**

Create `tests/test_library.py`:

```python
import json
from pathlib import Path
import pytest


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr("library._STORAGE_DIR", tmp_path)
    monkeypatch.setattr("library._REGISTRY_PATH", tmp_path / "sources.json")
    monkeypatch.setattr("library._DOCSTORE_DIR", tmp_path / "docstore")
    return tmp_path


class TestRegistry:
    def test_load_returns_empty_when_file_missing(self, storage):
        from library import load_registry
        assert load_registry() == []

    def test_save_then_load_round_trip(self, storage):
        from library import Source, save_registry, load_registry
        src = Source(
            source_id="abc-123",
            type="pdf",
            title="lec01.pdf",
            category="Day 1",
            chunk_count=42,
            metadata={"filename": "lec01.pdf"},
            added_at="2026-04-24T00:00:00Z",
        )
        save_registry([src])
        loaded = load_registry()
        assert len(loaded) == 1
        assert loaded[0].source_id == "abc-123"
        assert loaded[0].category == "Day 1"
        assert loaded[0].chunk_count == 42

    def test_save_creates_parent_dir(self, storage):
        from library import Source, save_registry
        src = Source(source_id="x", type="pdf", title="t", category="Other",
                     chunk_count=1, metadata={}, added_at="2026-04-24T00:00:00Z")
        save_registry([src])
        assert (storage / "sources.json").exists()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_library.py -v`
Expected: FAIL &mdash; `library` module not yet created.

- [ ] **Step 4: Implement `library.py` with `Source`, `load_registry`, `save_registry`**

Create `library.py`:

```python
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

_STORAGE_DIR = Path("storage")
_REGISTRY_PATH = _STORAGE_DIR / "sources.json"
_DOCSTORE_DIR = _STORAGE_DIR / "docstore"


@dataclass
class Source:
    source_id: str
    type: str                    # "pdf" | "youtube" | "github"
    title: str
    category: str
    chunk_count: int
    metadata: dict[str, Any]
    added_at: str                # ISO-8601 UTC


def load_registry() -> list[Source]:
    if not _REGISTRY_PATH.exists():
        return []
    raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    return [Source(**item) for item in raw]


def save_registry(sources: list[Source]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(s) for s in sources]
    _REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_library.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add library.py tests/test_library.py .gitignore
git commit -m "feat: add Source dataclass and sources.json registry"
```

---

## Task 2: Title derivation + `new_source` factory

**Files:**
- Modify: `library.py`
- Modify: `tests/test_library.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_library.py`:

```python
class TestNewSource:
    def test_pdf_title_is_filename(self, storage):
        from library import new_source
        src = new_source(type="pdf", category="Day 1", raw={"filename": "lec01.pdf"})
        assert src.type == "pdf"
        assert src.title == "lec01.pdf"
        assert src.category == "Day 1"
        assert src.source_id  # non-empty uuid
        assert src.added_at.endswith("Z")

    def test_youtube_title_is_url(self, storage):
        from library import new_source
        src = new_source(type="youtube", category="Other",
                         raw={"url": "https://youtu.be/xyz"})
        assert src.title == "https://youtu.be/xyz"
        assert src.metadata["url"] == "https://youtu.be/xyz"

    def test_github_title_is_owner_slash_repo(self, storage):
        from library import new_source
        src = new_source(type="github", category="Other",
                         raw={"owner": "anthropics", "repo": "cookbook"})
        assert src.title == "anthropics/cookbook"
        assert src.metadata["owner"] == "anthropics"
        assert src.metadata["repo"] == "cookbook"

    def test_source_ids_are_unique(self, storage):
        from library import new_source
        a = new_source(type="pdf", category="Other", raw={"filename": "a.pdf"})
        b = new_source(type="pdf", category="Other", raw={"filename": "b.pdf"})
        assert a.source_id != b.source_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_library.py::TestNewSource -v`
Expected: FAIL &mdash; `new_source` not defined.

- [ ] **Step 3: Implement `new_source` in `library.py`**

Append to `library.py`:

```python
import uuid
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _derive_title(type: str, raw: dict[str, Any]) -> str:
    if type == "pdf":
        return raw["filename"]
    if type == "youtube":
        return raw["url"]
    if type == "github":
        return f"{raw['owner']}/{raw['repo']}"
    raise ValueError(f"unknown source type: {type}")


def new_source(*, type: str, category: str, raw: dict[str, Any]) -> Source:
    return Source(
        source_id=str(uuid.uuid4()),
        type=type,
        title=_derive_title(type, raw),
        category=category,
        chunk_count=0,
        metadata=dict(raw),
        added_at=_now_iso(),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_library.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add library.py tests/test_library.py
git commit -m "feat: add new_source factory with per-type title derivation"
```

---

## Task 3: `tag_documents` &mdash; inject source_id + category into chunk metadata

**Files:**
- Modify: `library.py`
- Modify: `tests/test_library.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_library.py`:

```python
class TestTagDocuments:
    def test_attaches_source_id_and_category(self, storage):
        from library import tag_documents
        from llama_index.core import Document
        docs = [
            Document(text="a", metadata={"source": "pdf", "page": 1}),
            Document(text="b", metadata={"source": "pdf", "page": 2}),
        ]
        tag_documents(docs, source_id="sid-1", category="Day 1")
        for d in docs:
            assert d.metadata["source_id"] == "sid-1"
            assert d.metadata["category"] == "Day 1"

    def test_preserves_existing_metadata(self, storage):
        from library import tag_documents
        from llama_index.core import Document
        docs = [Document(text="a", metadata={"source": "pdf", "page": 7})]
        tag_documents(docs, source_id="sid-1", category="Other")
        assert docs[0].metadata["page"] == 7
        assert docs[0].metadata["source"] == "pdf"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_library.py::TestTagDocuments -v`
Expected: FAIL &mdash; `tag_documents` not defined.

- [ ] **Step 3: Implement `tag_documents`**

Append to `library.py`:

```python
from llama_index.core import Document


def tag_documents(docs: list[Document], *, source_id: str, category: str) -> None:
    """Mutate docs in place: add source_id + category to each metadata dict."""
    for d in docs:
        d.metadata["source_id"] = source_id
        d.metadata["category"] = category
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_library.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add library.py tests/test_library.py
git commit -m "feat: tag_documents injects source_id and category into chunks"
```

---

## Task 4: Per-source docstore (pickle save / load / delete)

**Files:**
- Modify: `library.py`
- Modify: `tests/test_library.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_library.py`:

```python
class TestDocstore:
    def test_save_then_load_round_trip(self, storage):
        from library import save_docs, load_docs
        from llama_index.core import Document
        docs = [
            Document(text="alpha", metadata={"source_id": "s1", "category": "Day 1"}),
            Document(text="beta",  metadata={"source_id": "s1", "category": "Day 1"}),
        ]
        save_docs("s1", docs)
        restored = load_docs("s1")
        assert len(restored) == 2
        assert restored[0].text == "alpha"
        assert restored[1].metadata["category"] == "Day 1"

    def test_delete_removes_pickle(self, storage):
        from library import save_docs, delete_docs, load_docs
        from llama_index.core import Document
        save_docs("s2", [Document(text="x", metadata={"source_id": "s2", "category": "Other"})])
        delete_docs("s2")
        assert load_docs("s2") == []

    def test_load_missing_returns_empty(self, storage):
        from library import load_docs
        assert load_docs("nonexistent") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_library.py::TestDocstore -v`
Expected: FAIL.

- [ ] **Step 3: Implement pickle helpers**

Append to `library.py`:

```python
import pickle


def _docstore_path(source_id: str) -> Path:
    return _DOCSTORE_DIR / f"{source_id}.pkl"


def save_docs(source_id: str, docs: list[Document]) -> None:
    _DOCSTORE_DIR.mkdir(parents=True, exist_ok=True)
    with _docstore_path(source_id).open("wb") as fh:
        pickle.dump(docs, fh)


def load_docs(source_id: str) -> list[Document]:
    path = _docstore_path(source_id)
    if not path.exists():
        return []
    with path.open("rb") as fh:
        return pickle.load(fh)


def delete_docs(source_id: str) -> None:
    path = _docstore_path(source_id)
    if path.exists():
        path.unlink()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_library.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add library.py tests/test_library.py
git commit -m "feat: per-source docstore pickles (save/load/delete)"
```

---

## Task 5: Rewrite `vector_store.py` for persistent + incremental index

**Files:**
- Modify: `vector_store.py`
- Create: `tests/test_vector_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vector_store.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from llama_index.core import Document, Settings
from llama_index.core.embeddings.mock_embed_model import MockEmbedding


@pytest.fixture(autouse=True)
def mock_embeddings():
    Settings.embed_model = MockEmbedding(embed_dim=8)
    yield


@pytest.fixture
def chroma_dir(tmp_path, monkeypatch):
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
        from vector_store import load_or_create_index, insert_documents, delete_source, _collection_size
        index = load_or_create_index()
        insert_documents(index, [_doc("a", "s1"), _doc("b", "s1"), _doc("c", "s2")])
        assert _collection_size() == 3
        delete_source(index, "s1")
        assert _collection_size() == 1

    def test_load_rehydrates_existing_collection(self, chroma_dir):
        from vector_store import load_or_create_index, insert_documents, _collection_size
        index1 = load_or_create_index()
        insert_documents(index1, [_doc("persisted", "s1")])
        # Simulate a fresh process: new index instance on same chroma dir.
        index2 = load_or_create_index()
        assert _collection_size() == 1
        assert index2 is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vector_store.py -v`
Expected: FAIL &mdash; functions don't exist yet.

- [ ] **Step 3: Rewrite `vector_store.py`**

Replace entire contents of `vector_store.py`:

```python
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore

_CHROMA_PATH = "./chroma_db"
_COLLECTION_NAME = "learning_materials"


def _client():
    return chromadb.PersistentClient(path=_CHROMA_PATH)


def _collection():
    return _client().get_or_create_collection(_COLLECTION_NAME)


def _collection_size() -> int:
    return _collection().count()


def load_or_create_index() -> VectorStoreIndex:
    """Load existing collection as an index, or create an empty one."""
    collection = _collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    if collection.count() == 0:
        return VectorStoreIndex.from_documents([], storage_context=storage_context)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vector_store.py -v`
Expected: all 4 pass.

- [ ] **Step 5: Confirm existing ingestion tests still pass**

Run: `pytest tests/test_ingestion.py -v`
Expected: all pass (no changes to ingestion.py).

- [ ] **Step 6: Commit**

```bash
git add vector_store.py tests/test_vector_store.py
git commit -m "feat: persistent incremental vector store (load/insert/delete)"
```

---

## Task 6: Orchestrator &mdash; `ingest_and_register` + `remove_source`

**Files:**
- Modify: `library.py`
- Modify: `tests/test_library.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_library.py`:

```python
class TestIngestAndRegister:
    def test_ingest_and_register_happy_path(self, storage, monkeypatch):
        from library import ingest_and_register, load_registry, load_docs
        from llama_index.core import Document
        pdf_doc = Document(text="page1", metadata={"source": "pdf", "page": 1})
        monkeypatch.setattr("library.ingest", lambda **kwargs: [pdf_doc])

        fake_index = object()
        insert_calls = []
        def fake_insert(index, docs):
            insert_calls.append((index, docs))
            return len(docs)
        monkeypatch.setattr("library.insert_documents", fake_insert)

        src = ingest_and_register(
            index=fake_index,
            category="Day 1",
            pdf_bytes=b"fake",
            pdf_filename="lec01.pdf",
            youtube_url=None,
            github_url=None,
        )
        assert src.type == "pdf"
        assert src.title == "lec01.pdf"
        assert src.category == "Day 1"
        assert src.chunk_count == 1
        assert insert_calls[0][0] is fake_index
        assert insert_calls[0][1][0].metadata["source_id"] == src.source_id
        assert insert_calls[0][1][0].metadata["category"] == "Day 1"
        assert load_docs(src.source_id)[0].text == "page1"
        assert load_registry()[0].source_id == src.source_id

    def test_ingest_and_register_rejects_empty_inputs(self, storage, monkeypatch):
        from library import ingest_and_register
        monkeypatch.setattr("library.ingest", lambda **kwargs: [])
        with pytest.raises(ValueError, match="no content"):
            ingest_and_register(
                index=object(), category="Other",
                pdf_bytes=None, pdf_filename=None,
                youtube_url=None, github_url=None,
            )


class TestRemoveSource:
    def test_remove_source_deletes_everything(self, storage, monkeypatch):
        from library import save_registry, save_docs, remove_source, load_registry, load_docs, Source
        from llama_index.core import Document
        src = Source(source_id="gone", type="pdf", title="t", category="Other",
                     chunk_count=1, metadata={}, added_at="2026-04-24T00:00:00Z")
        save_registry([src])
        save_docs("gone", [Document(text="x", metadata={})])

        delete_calls = []
        monkeypatch.setattr("library.delete_source_vectors",
                            lambda index, sid: delete_calls.append(sid))

        remove_source(index=object(), source_id="gone")
        assert load_registry() == []
        assert load_docs("gone") == []
        assert delete_calls == ["gone"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_library.py::TestIngestAndRegister tests/test_library.py::TestRemoveSource -v`
Expected: FAIL.

- [ ] **Step 3: Implement orchestrator**

Append to `library.py`:

```python
from ingestion import ingest
from vector_store import insert_documents, delete_source as delete_source_vectors


def ingest_and_register(
    *,
    index,
    category: str,
    pdf_bytes: bytes | None,
    pdf_filename: str | None,
    youtube_url: str | None,
    github_url: str | None,
) -> Source:
    """Load -> tag -> pickle -> insert into index -> append registry. Atomic-ish.

    Exactly one of the three inputs should be supplied per call (UI enforces this).
    """
    docs = ingest(
        pdf_bytes=pdf_bytes,
        youtube_url=youtube_url,
        github_url=github_url,
    )
    if not docs:
        raise ValueError("no content could be extracted from the provided input")

    if pdf_bytes is not None:
        src = new_source(type="pdf", category=category, raw={"filename": pdf_filename or "upload.pdf"})
    elif youtube_url:
        src = new_source(type="youtube", category=category, raw={"url": youtube_url})
    else:
        from ingestion import _parse_github_owner_repo
        owner, repo = _parse_github_owner_repo(github_url)
        src = new_source(type="github", category=category, raw={"owner": owner, "repo": repo})

    tag_documents(docs, source_id=src.source_id, category=category)
    save_docs(src.source_id, docs)
    chunk_count = insert_documents(index, docs)

    src = Source(**{**asdict(src), "chunk_count": chunk_count})
    registry = load_registry()
    registry.append(src)
    save_registry(registry)
    return src


def remove_source(*, index, source_id: str) -> None:
    delete_source_vectors(index, source_id)
    delete_docs(source_id)
    registry = [s for s in load_registry() if s.source_id != source_id]
    save_registry(registry)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_library.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add library.py tests/test_library.py
git commit -m "feat: ingest_and_register + remove_source orchestrators"
```

---

## Task 7: Scope helpers &mdash; `list_categories` + `get_docs_for_scope`

**Files:**
- Modify: `library.py`
- Modify: `tests/test_library.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_library.py`:

```python
class TestScopeHelpers:
    def test_list_categories_returns_defaults_when_empty(self, storage):
        from library import list_categories
        assert list_categories() == ["Day 1", "Day 2", "Day 3", "Other"]

    def test_list_categories_union_with_existing(self, storage):
        from library import Source, save_registry, list_categories
        save_registry([
            Source(source_id="a", type="pdf", title="t", category="Week 1",
                   chunk_count=0, metadata={}, added_at="2026-04-24T00:00:00Z"),
        ])
        cats = list_categories()
        assert "Week 1" in cats
        assert "Day 1" in cats
        assert "Other" in cats

    def test_get_docs_for_scope_all(self, storage):
        from library import Source, save_registry, save_docs, get_docs_for_scope
        from llama_index.core import Document
        save_registry([
            Source(source_id="s1", type="pdf", title="a", category="Day 1",
                   chunk_count=1, metadata={}, added_at="2026-04-24T00:00:00Z"),
            Source(source_id="s2", type="pdf", title="b", category="Day 2",
                   chunk_count=1, metadata={}, added_at="2026-04-24T00:00:00Z"),
        ])
        save_docs("s1", [Document(text="alpha", metadata={})])
        save_docs("s2", [Document(text="beta",  metadata={})])
        docs = get_docs_for_scope("All")
        assert sorted(d.text for d in docs) == ["alpha", "beta"]

    def test_get_docs_for_scope_category(self, storage):
        from library import Source, save_registry, save_docs, get_docs_for_scope
        from llama_index.core import Document
        save_registry([
            Source(source_id="s1", type="pdf", title="a", category="Day 1",
                   chunk_count=1, metadata={}, added_at="2026-04-24T00:00:00Z"),
            Source(source_id="s2", type="pdf", title="b", category="Day 2",
                   chunk_count=1, metadata={}, added_at="2026-04-24T00:00:00Z"),
        ])
        save_docs("s1", [Document(text="alpha", metadata={})])
        save_docs("s2", [Document(text="beta",  metadata={})])
        docs = get_docs_for_scope("Day 1")
        assert [d.text for d in docs] == ["alpha"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_library.py::TestScopeHelpers -v`
Expected: FAIL.

- [ ] **Step 3: Implement helpers**

Append to `library.py`:

```python
_DEFAULT_CATEGORIES = ["Day 1", "Day 2", "Day 3", "Other"]


def list_categories() -> list[str]:
    """Union of default categories and categories currently present in the registry.

    Returned in a stable order: defaults first (in defined order), then any
    user-added categories in insertion order.
    """
    seen = list(_DEFAULT_CATEGORIES)
    for s in load_registry():
        if s.category not in seen:
            seen.append(s.category)
    return seen


def get_docs_for_scope(scope: str) -> list[Document]:
    """Return the Document list matching the scope.

    scope == "All"   -> every source's pickled docs concatenated
    scope == "<cat>" -> only sources whose category == scope
    """
    registry = load_registry()
    if scope != "All":
        registry = [s for s in registry if s.category == scope]
    docs: list[Document] = []
    for s in registry:
        docs.extend(load_docs(s.source_id))
    return docs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_library.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add library.py tests/test_library.py
git commit -m "feat: list_categories and get_docs_for_scope helpers"
```

---

## Task 8: Scope-aware chat engine

**Files:**
- Modify: `chat.py`
- Create: `tests/test_chat_scope.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chat_scope.py`:

```python
from unittest.mock import MagicMock, patch


class TestCreateChatEngine:
    def test_scope_all_passes_no_filter(self):
        from chat import create_chat_engine
        index = MagicMock()
        index.as_retriever = MagicMock()
        create_chat_engine(index, scope="All")
        kwargs = index.as_retriever.call_args.kwargs
        assert kwargs.get("filters") is None
        assert kwargs["similarity_top_k"] == 5

    def test_scope_category_adds_metadata_filter(self):
        from chat import create_chat_engine
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
        index = MagicMock()
        index.as_retriever = MagicMock()
        create_chat_engine(index, scope="Day 1")
        filters = index.as_retriever.call_args.kwargs["filters"]
        assert isinstance(filters, MetadataFilters)
        assert len(filters.filters) == 1
        f = filters.filters[0]
        assert f.key == "category"
        assert f.value == "Day 1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chat_scope.py -v`
Expected: FAIL &mdash; `create_chat_engine` signature doesn't accept `scope`.

- [ ] **Step 3: Update `chat.py`**

Replace the contents of `chat.py`:

```python
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter


def create_chat_engine(index, scope: str = "All") -> CondensePlusContextChatEngine:
    kwargs = {"similarity_top_k": 5, "filters": None}
    if scope != "All":
        kwargs["filters"] = MetadataFilters(
            filters=[MetadataFilter(key="category", value=scope)]
        )
    retriever = index.as_retriever(**kwargs)
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

Run: `pytest tests/test_chat_scope.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add chat.py tests/test_chat_scope.py
git commit -m "feat: scope-aware chat engine with metadata filter"
```

---

## Task 9: Left pane &mdash; `ui/source_library.py`

**Files:**
- Create: `ui/__init__.py`
- Create: `ui/source_library.py`

Streamlit UI is manually verified (no unit tests). This module is a single pure function called by `app.py`.

- [ ] **Step 1: Create empty `ui/__init__.py`**

Create `ui/__init__.py` with no content (just makes `ui` a package).

- [ ] **Step 2: Implement `ui/source_library.py`**

Create `ui/source_library.py`:

```python
"""Left pane: grouped source list + inline Add Source form."""
import streamlit as st

from library import (
    Source,
    list_categories,
    ingest_and_register,
    remove_source,
)

_TYPE_ICON = {"pdf": "📄", "youtube": "🎥", "github": "💻"}


def render(registry: list[Source], index) -> None:
    """Render the source library panel. Mutates st.session_state on add/remove."""
    st.markdown("### Sources")
    total = len(registry)
    cats = {s.category for s in registry}
    st.caption(f"{total} item{'s' if total != 1 else ''} · {len(cats)} categor{'ies' if len(cats) != 1 else 'y'}")

    if st.button("+ Add Source", use_container_width=True, key="toggle_add_form"):
        st.session_state.add_form_open = not st.session_state.get("add_form_open", False)

    if st.session_state.get("add_form_open"):
        _render_add_form(index)

    _render_grouped_list(registry, index)


def _render_add_form(index) -> None:
    with st.container(border=True):
        st.markdown("**Add a source**")
        pdf = st.file_uploader("PDF", type=["pdf"], key="add_pdf")
        yt = st.text_input("YouTube URL", key="add_yt",
                           placeholder="https://www.youtube.com/watch?v=...")
        gh = st.text_input("GitHub URL", key="add_gh",
                           placeholder="https://github.com/owner/repo")
        cat = st.selectbox("Category", options=list_categories(), index=3, key="add_cat")

        cols = st.columns(2)
        if cols[0].button("Add", type="primary", use_container_width=True, key="add_submit"):
            try:
                src = ingest_and_register(
                    index=index,
                    category=cat,
                    pdf_bytes=pdf.read() if pdf else None,
                    pdf_filename=pdf.name if pdf else None,
                    youtube_url=yt.strip() or None,
                    github_url=gh.strip() or None,
                )
                st.session_state.registry.append(src)
                st.session_state.add_form_open = False
                st.session_state.chat_engine = None      # force rebuild
                st.session_state.outline_cache = {}      # invalidate cached generations
                st.session_state.exercises_cache = {}
                st.success(f"Added: {src.title}")
                st.rerun()
            except Exception as e:
                st.error(f"Add failed: {e}")
        if cols[1].button("Cancel", use_container_width=True, key="add_cancel"):
            st.session_state.add_form_open = False
            st.rerun()


def _render_grouped_list(registry: list[Source], index) -> None:
    if not registry:
        st.info("Add your first source to begin.")
        return

    by_cat: dict[str, list[Source]] = {}
    for s in registry:
        by_cat.setdefault(s.category, []).append(s)

    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        with st.expander(f"{cat} · {len(items)}", expanded=True):
            for s in items:
                _render_card(s, index)


def _render_card(s: Source, index) -> None:
    icon = _TYPE_ICON.get(s.type, "📦")
    col_title, col_remove = st.columns([8, 1])
    col_title.markdown(f"{icon} **{s.title}**  \n<span style='color:#888;font-size:11px'>{s.chunk_count} chunks</span>",
                       unsafe_allow_html=True)
    if col_remove.button("🗑", key=f"rm_{s.source_id}", help="Remove source"):
        try:
            remove_source(index=index, source_id=s.source_id)
            st.session_state.registry = [r for r in st.session_state.registry
                                         if r.source_id != s.source_id]
            st.session_state.chat_engine = None
            st.session_state.outline_cache = {}
            st.session_state.exercises_cache = {}
            st.rerun()
        except Exception as e:
            st.error(f"Remove failed: {e}")
```

- [ ] **Step 3: Syntax / import check**

Run: `python -c "import ui.source_library"`
Expected: no output (imports cleanly).

- [ ] **Step 4: Commit**

```bash
git add ui/__init__.py ui/source_library.py
git commit -m "feat: source library UI panel with grouped cards and Add form"
```

---

## Task 10: Right pane &mdash; `ui/workspace.py`

**Files:**
- Create: `ui/workspace.py`

- [ ] **Step 1: Implement `ui/workspace.py`**

Create `ui/workspace.py`:

```python
"""Right pane: scope selector + Chat / Outline / Exercises tabs."""
import streamlit as st

from library import Source, get_docs_for_scope
from chat import create_chat_engine, chat
from outline import generate_outline
from exercises import generate_exercises


def render(registry: list[Source], index) -> None:
    if not registry:
        st.info("Add a source on the left to start chatting.")
        return

    _render_scope_selector(registry)
    tab_chat, tab_outline, tab_exercises = st.tabs(["💬 Chat", "📋 Outline", "🏋️ Exercises"])
    with tab_chat:
        _render_chat(index)
    with tab_outline:
        _render_outline()
    with tab_exercises:
        _render_exercises()


def _render_scope_selector(registry: list[Source]) -> None:
    cats = sorted({s.category for s in registry})
    options = ["All"] + cats
    current = st.session_state.get("scope", "All")
    if current not in options:
        current = "All"
    new_scope = st.selectbox(
        "Scope",
        options=options,
        index=options.index(current),
        key="scope_selector",
    )
    if new_scope != current:
        st.session_state.scope = new_scope
        st.session_state.chat_engine = None
        st.session_state.messages = []
        st.rerun()
    st.session_state.scope = new_scope


def _render_chat(index) -> None:
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
            st.session_state.chat_engine = create_chat_engine(
                index, scope=st.session_state.scope
            )

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = chat(st.session_state.chat_engine, user_input)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


def _render_outline() -> None:
    scope = st.session_state.scope
    cache = st.session_state.outline_cache
    if st.button("Generate Outline", key="gen_outline"):
        docs = get_docs_for_scope(scope)
        if not docs:
            st.warning("No documents in the current scope.")
        else:
            with st.spinner("Generating outline…"):
                try:
                    cache[scope] = generate_outline(docs)
                except Exception as e:
                    st.error(f"Outline generation failed: {e}")
    if cache.get(scope):
        st.markdown(cache[scope])


def _render_exercises() -> None:
    scope = st.session_state.scope
    cache = st.session_state.exercises_cache
    if st.button("Generate Exercises", key="gen_exercises"):
        docs = get_docs_for_scope(scope)
        if not docs:
            st.warning("No documents in the current scope.")
        else:
            with st.spinner("Generating exercises…"):
                try:
                    cache[scope] = generate_exercises(docs)
                except Exception as e:
                    st.error(f"Exercise generation failed: {e}")
    if cache.get(scope):
        st.markdown(cache[scope])
```

- [ ] **Step 2: Syntax / import check**

Run: `python -c "import ui.workspace"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add ui/workspace.py
git commit -m "feat: workspace UI panel with scope selector and tabs"
```

---

## Task 11: Rewrite `app.py` as two-column layout

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace `app.py` entirely**

Replace the contents of `app.py`:

```python
import streamlit as st
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

from library import load_registry
from vector_store import load_or_create_index
from ui.source_library import render as render_library
from ui.workspace import render as render_workspace

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

st.set_page_config(page_title="AI Learning Assistant", layout="wide",
                   initial_sidebar_state="collapsed")
st.title("AI Learning Assistant")


@st.cache_resource
def _get_index():
    return load_or_create_index()


_STATE_DEFAULTS = {
    "registry": None,                 # list[Source], loaded on first run
    "scope": "All",
    "chat_engine": None,
    "messages": [],
    "outline_cache": {},              # {scope: str}
    "exercises_cache": {},            # {scope: str}
    "add_form_open": False,
}
for key, default in _STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.registry is None:
    st.session_state.registry = load_registry()

index = _get_index()

col_lib, col_work = st.columns([1, 3], gap="medium")
with col_lib:
    render_library(st.session_state.registry, index)
with col_work:
    render_workspace(st.session_state.registry, index)
```

- [ ] **Step 2: Syntax check**

Run: `python -c "import ast; ast.parse(open('app.py').read())"`
Expected: no output.

- [ ] **Step 3: Full test suite**

Run: `pytest -v`
Expected: all tests pass (ingestion + library + vector_store + chat_scope).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: two-column NotebookLM-style layout in app.py"
```

---

## Task 12: One-time legacy Chroma wipe + manual smoke test

**Files:**
- No code changes; runtime + manual verification.

- [ ] **Step 1: Wipe legacy Chroma state (if any)**

The old `vector_store.py` always deleted+recreated the collection on every `build_index`, so chunks from previous runs never persisted. If a `chroma_db/` directory exists from earlier development, its contents have no `source_id` / `category` metadata and would be orphaned. Wipe it:

```bash
rm -rf chroma_db storage
```

(The directories will be recreated automatically on next app start.)

- [ ] **Step 2: Launch the app**

Run: `streamlit run app.py`
Expected: browser opens; title shows, left column shows "Sources · 0 items" with an "Add your first source to begin." note; right column shows "Add a source on the left to start chatting."

- [ ] **Step 3: Smoke-test &mdash; add a PDF under "Day 1"**

1. Click **+ Add Source**.
2. Upload a short PDF. Pick category `Day 1`. Click **Add**.
3. Expect: success toast "Added: &lt;filename&gt;", form collapses, a card appears under the `Day 1 · 1` group showing chunk count.
4. Inspect: `storage/sources.json` should contain one entry; `storage/docstore/*.pkl` should have one file.

- [ ] **Step 4: Smoke-test &mdash; add a second source under "Day 2"**

1. Click **+ Add Source** again.
2. Paste any short YouTube URL with an English transcript. Pick `Day 2`. Click **Add**.
3. Expect: card appears under a new `Day 2 · 1` expander.

- [ ] **Step 5: Smoke-test &mdash; chat scope filtering**

1. Scope = `All`: ask "What is this about?" &rarr; answer references both sources.
2. Switch scope to `Day 1` &rarr; conversation resets (expected per spec).
3. Ask the same question &rarr; answer only references the PDF content.

- [ ] **Step 6: Smoke-test &mdash; outline & exercises caching**

1. Scope = `All`; click **Generate Outline** &rarr; content appears.
2. Switch to `Day 1`, switch back to `All`: outline is still shown (cache hit, no regeneration).
3. Remove a source &rarr; switching to the affected scope and clicking Generate regenerates fresh.

- [ ] **Step 7: Smoke-test &mdash; persistence across restarts**

1. Stop the streamlit process.
2. Re-launch `streamlit run app.py`.
3. Expect: both source cards visible; scope selector retains categories; chat works against the same corpus without re-ingesting.

- [ ] **Step 8: Smoke-test &mdash; remove source**

1. Click the 🗑 icon on one of the cards.
2. Expect: card disappears; card count decrements; `storage/sources.json` no longer lists that `source_id`; `storage/docstore/<id>.pkl` is deleted.

- [ ] **Step 9: Record smoke-test outcome**

Write a brief "done / issues found" note in `docs/superpowers/plans/2026-04-24-notebooklm-layout-smoke-test.md` if any issues surface. If all steps pass, no file needed.

- [ ] **Step 10: Final commit (if anything was tweaked during smoke test)**

```bash
git status
git add -p  # review carefully
git commit -m "chore: fixes from smoke test" # only if something changed
```

---

## Done When

- All unit tests pass: `pytest -v`
- `streamlit run app.py` loads without errors and shows the two-pane layout.
- Steps 3&ndash;8 of Task 12 all pass as described.
- Registry + chroma survive a process restart.
- Removing a source deletes its chunks and pickle and registry entry.
