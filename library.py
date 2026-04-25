import json
import pickle
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_index.core import Document

from ingestion import ingest
from vector_store import insert_documents, delete_source as delete_source_vectors

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


def tag_documents(docs: list[Document], *, source_id: str, category: str) -> None:
    """Mutate docs in place: add source_id + category to each metadata dict."""
    for d in docs:
        d.metadata["source_id"] = source_id
        d.metadata["category"] = category


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
    try:
        chunk_count = insert_documents(index, docs)
    except Exception:
        # Compensate so a mid-insert failure does not leave orphaned chunks
        # in Chroma or an orphan pickle on disk.
        delete_docs(src.source_id)
        try:
            delete_source_vectors(index, src.source_id)
        except Exception:
            pass
        raise

    src = Source(**{**asdict(src), "chunk_count": chunk_count})
    registry = load_registry()
    registry.append(src)
    save_registry(registry)
    return src


def remove_source(*, index, source_id: str) -> None:
    # Order matters: chroma -> registry -> pickle. If pickle unlink fails
    # (Windows file lock, etc.) the registry is already consistent with
    # chroma; the orphan pickle is harmless and can be cleaned manually.
    delete_source_vectors(index, source_id)
    registry = [s for s in load_registry() if s.source_id != source_id]
    save_registry(registry)
    try:
        delete_docs(source_id)
    except OSError:
        pass


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
