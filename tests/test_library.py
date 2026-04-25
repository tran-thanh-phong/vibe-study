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

    def test_ingest_and_register_compensates_on_insert_failure(self, storage, monkeypatch):
        from library import ingest_and_register, load_registry, load_docs
        from llama_index.core import Document
        pdf_doc = Document(text="page1", metadata={"source": "pdf", "page": 1})
        monkeypatch.setattr("library.ingest", lambda **kwargs: [pdf_doc])

        def boom(index, docs):
            raise RuntimeError("embedding service unavailable")
        monkeypatch.setattr("library.insert_documents", boom)

        delete_calls = []
        monkeypatch.setattr("library.delete_source_vectors",
                            lambda index, sid: delete_calls.append(sid))

        with pytest.raises(RuntimeError, match="embedding service"):
            ingest_and_register(
                index=object(), category="Day 1",
                pdf_bytes=b"fake", pdf_filename="lec01.pdf",
                youtube_url=None, github_url=None,
            )
        # Registry must not mention the half-ingested source.
        assert load_registry() == []
        # Pickle for the attempted source must be cleaned up — scan docstore dir.
        pkl_files = list((storage / "docstore").glob("*.pkl")) if (storage / "docstore").exists() else []
        assert pkl_files == []
        # Compensating chroma wipe must be attempted once.
        assert len(delete_calls) == 1


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

    def test_remove_source_tolerates_pickle_unlink_failure(self, storage, monkeypatch):
        from library import save_registry, save_docs, remove_source, load_registry, Source
        from llama_index.core import Document
        src = Source(source_id="locked", type="pdf", title="t", category="Other",
                     chunk_count=1, metadata={}, added_at="2026-04-25T00:00:00Z")
        save_registry([src])
        save_docs("locked", [Document(text="x", metadata={})])

        monkeypatch.setattr("library.delete_source_vectors", lambda index, sid: None)
        def fail_delete(_sid):
            raise OSError("file in use")
        monkeypatch.setattr("library.delete_docs", fail_delete)

        # Should not raise — pickle unlink failure is swallowed.
        remove_source(index=object(), source_id="locked")
        # Registry MUST be saved (so UI no longer shows the source).
        assert load_registry() == []


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
