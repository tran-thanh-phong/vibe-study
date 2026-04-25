# NotebookLM-Style Source Library &mdash; Design Spec

**Date:** 2026-04-24
**Status:** Approved (direction), pending user review of this spec
**Supersedes (UI layer only):** `2026-04-24-ai-learning-assistant-design.md` &mdash; core ingestion/RAG logic carries over; this spec replaces the single-shot sidebar-ingest UX with an incremental, categorised source library.

---

## 1. Goal

Replace the current single-shot upload flow (sidebar inputs &rarr; "Process Content" rebuilds the whole index &rarr; 3 tabs) with a NotebookLM-inspired two-pane layout:

- **Left pane:** a persistent, categorised source library the user builds up over time. Sources are grouped by user-assigned categories (e.g. `Day 1`, `Day 2`, `Other`). Each source card shows type, title, and chunk count so the user can finally see what has been ingested.
- **Right pane:** Chat / Outline / Exercises tabs (same three existing features) with a scope selector that limits generation to one category or all sources.

The driving motivation is the user's complaint: *"we don't [know] how many source files we ingest"*. The fix is to make the library visible, addressable, and incremental.

## 2. Non-goals

- Auth, multi-user, or multi-notebook (course) management &mdash; still one global library per install.
- Re-designing ingestion pipelines (PDF / YouTube / GitHub loaders are untouched).
- Changing the RAG strategy, embedding model, or LLM.
- Chunk-level citations in chat replies (retained as a future enhancement).

## 3. Key user-visible changes

| Before | After |
|---|---|
| Sidebar contains all three inputs + one "Process Content" button | Left pane shows the library; an inline "+ Add Source" form replaces the sidebar flow |
| Processing a new source wipes the previous index | Sources accumulate; each add inserts into the existing index |
| No way to see what was ingested | Every source is listed as a card (icon, title, chunk count, category) |
| No way to remove a single source | Each card has a Remove action that deletes its chunks from Chroma |
| Outline / Exercises / Chat always run over everything | A scope selector (`All sources` / `Day 1` / &hellip;) constrains retrieval and generation |
| Index is in-memory only; lost on restart | Index and source registry persist to disk; library survives Streamlit reruns and restarts |

## 4. Data model

### 4.1 Source

A **source** is one user-added input (one PDF, one YouTube URL, one GitHub repo). Each source produces N chunks internally, but the UI treats it as a single unit.

```python
# storage/sources.json  (one array; file is the source-of-truth registry)
{
  "source_id": "uuid4 string",
  "type": "pdf" | "youtube" | "github",
  "title": "lec01-slides.pdf" | "What is RAG? (video)" | "owner/repo",
  "category": "Day 1",               # user-chosen, free-text
  "chunk_count": 42,
  "metadata": {                       # type-specific original inputs
    "filename": "lec01-slides.pdf",   # pdf
    "url": "https://youtu.be/...",    # youtube
    "owner": "anthropics",            # github
    "repo": "anthropic-cookbook"
  },
  "added_at": "2026-04-24T12:34:56Z"
}
```

### 4.2 Chunk metadata (in Chroma)

Every chunk already carries `source` (pdf/youtube/github). We add two more fields so we can filter by source and category at retrieval time:

```python
Document.metadata = {
  "source": "pdf",          # existing
  "page": 3,                # existing (pdf only)
  "source_id": "<uuid>",    # NEW &mdash; enables per-source deletion
  "category": "Day 1"       # NEW &mdash; enables scope filtering
}
```

### 4.3 Category

Free-text string chosen by the user from a dropdown at upload. Dropdown options = union of (`Day 1`, `Day 2`, `Day 3`, `Other`) and all existing categories currently in the registry, plus a `+ New category&hellip;` option. Default selection: `Other`.

No category hierarchy, no reassignment after upload (for MVP &mdash; can be added later by editing `sources.json`).

### 4.3.1 Title derivation

Titles are computed at ingest time from data we already have on hand &mdash; no extra network calls for MVP:

| Type | Title |
|---|---|
| pdf | `UploadedFile.name` (Streamlit provides this) |
| youtube | the raw URL (a future enhancement can fetch the real video title via YouTube oEmbed) |
| github | `"<owner>/<repo>"` (already parsed by `_parse_github_owner_repo`) |

### 4.4 Persistence layout

```
chroma_db/                      # Chroma vector store (existing path, now persisted across runs)
  └── ... (chromadb internals)
storage/
  ├── sources.json              # the source registry (array of Source records)
  └── docstore/
      └── <source_id>.pkl       # pickled list[Document] per source
                                # needed because outline/exercises consume Documents, not chunks
```

The `storage/docstore/<source_id>.pkl` files let us reconstruct the `list[Document]` for a scoped outline/exercises run without having to re-download PDFs/transcripts/repos.

## 5. Architecture

### 5.1 Module layout (what changes, what's new)

```
app.py                  MODIFIED  two-pane layout, wires new modules
ingestion.py            MODIFIED  attach source_id + category metadata to every Document; return title info
vector_store.py         REWRITTEN incremental add / remove; persistent load
chat.py                 MODIFIED  scope-aware retriever (metadata filter)
outline.py              MODIFIED  accept scoped Document list
exercises.py            MODIFIED  accept scoped Document list
library.py              NEW       source registry: load, save, add, remove, list
ui/source_library.py    NEW       left-pane rendering (grouped list + Add Source form)
ui/workspace.py         NEW       right-pane rendering (tabs + scope selector)
```

200-line cap per file is respected by the split.

### 5.2 Key module contracts

**`library.py`** &mdash; single source of truth for the source registry.
```python
def load_registry() -> list[Source]: ...
def add_source(source: Source, docs: list[Document]) -> None:  # writes sources.json + pickle
def remove_source(source_id: str) -> None:                      # deletes pickle, updates sources.json
def list_categories() -> list[str]:                             # for the dropdown
def get_docs(source_ids: list[str]) -> list[Document]:          # for outline/exercises scoping
```

**`vector_store.py`** &mdash; persistent, incremental index.
```python
def load_or_create_index() -> VectorStoreIndex:      # called once on app start; rehydrates from chroma_db
def insert_documents(index, docs: list[Document]) -> int:  # returns chunk count
def delete_source(index, source_id: str) -> None:    # chroma where={"source_id": source_id}
```

**`chat.py`** &mdash; scope-aware retriever.
```python
def create_chat_engine(index, scope: Scope) -> ChatEngine:
    # scope = ALL  &rarr; no filter
    # scope = category("Day 1") &rarr; MetadataFilter(key="category", value="Day 1")
```

### 5.3 Session state

```python
st.session_state = {
    "index":    VectorStoreIndex,     # loaded once, mutated by add/remove
    "registry": list[Source],         # mirrors sources.json; re-read after mutations
    "scope":    "All" | "<category>", # drives chat engine + outline/exercises
    "chat_engine":  ChatEngine | None,# invalidated when scope changes
    "messages": list[dict],
    "outline":  str | None,           # keyed by scope; cleared on scope change
    "exercises":str | None,           # same
    "add_form_open": bool,            # toggles the inline Add Source form
}
```

## 6. UX flows

### 6.1 First launch (empty library)
- Left pane: header "Sources &middot; 0 items", big "+ Add Source" button, empty-state text: *"Add your first source to begin."*
- Right pane: disabled tabs + placeholder *"Add a source on the left to start chatting."*

### 6.2 Add a source
1. User clicks **+ Add Source** &rarr; a collapsible form (`st.session_state.add_form_open` toggles an expander-style block) reveals itself directly below the button in the left pane.
2. Form fields: PDF uploader, YouTube URL, GitHub URL, **Category** dropdown (existing categories + `+ New&hellip;`), **Add** button.
3. On submit: run `ingest(...)`, tag each Document with `source_id`+`category`, call `insert_documents`, write pickle + append to `sources.json`.
4. Form collapses; new card appears under its category group.
5. Chat engine is invalidated (rebuilt lazily on next message) so the new source is retrievable immediately.

### 6.3 Remove a source
1. User clicks trash icon on a card &rarr; confirmation toast.
2. On confirm: `delete_source(index, source_id)` (Chroma metadata-filtered delete), delete pickle, update registry.
3. Card disappears; chat engine invalidated; outline/exercises for that scope cleared.

### 6.4 Switch scope
1. User picks a category from the scope selector on the right pane.
2. `st.session_state.scope` updates; chat engine, outline, exercises are all invalidated.
3. Subsequent chat messages retrieve only from the scoped subset; outline/exercises regenerate on next click against the scoped docs.

## 7. Persistence &amp; migration

- **First run after this change:** existing `chroma_db/` (if any) is wiped once via a one-line "legacy collection without `source_id` metadata found &mdash; resetting" check at startup, because old chunks have no `source_id` and would be orphaned. Acceptable for MVP because the app is a local demo with no real user data.
- **Normal runs:** `chroma_db/` and `storage/` persist. The index and registry survive Streamlit reruns and process restarts. No server-side caching needed beyond what Chroma already does.

## 8. Error handling

- Ingestion failure (bad PDF, private repo, unavailable transcript): source is **not** written to registry or index; user gets a toast with the error. Same behaviour as today.
- Index/registry drift (a `source_id` in `sources.json` has no chunks in Chroma, or vice versa): detect on startup, log a warning, and self-heal by removing orphan registry entries. Orphan chunks (chunks whose `source_id` is missing from registry) are ignored, not deleted, to avoid accidental data loss.
- Chroma delete partial failure: surface the error; registry is updated only if delete succeeds.

## 9. Testing

Existing tests in `tests/test_ingestion.py` stay green &mdash; `ingest()` signature gains optional category arg but defaults to `Other`. New tests:

- `tests/test_library.py` &mdash; add/remove/list-categories round-trip, persistence across reload.
- `tests/test_vector_store.py` &mdash; `insert_documents` + `delete_source` leave index consistent; `load_or_create_index` rehydrates correctly.
- `tests/test_chat_scope.py` &mdash; chat engine with `scope=category("X")` only surfaces chunks from X.

UI (Streamlit) stays manual-test &mdash; smoke test checklist in the plan.

## 10. Risks

| Risk | Mitigation |
|---|---|
| ChromaDB metadata-filtered delete semantics vary by version | Pin `chromadb` version; add a test asserting post-delete chunk count |
| Pickle docstore is brittle across llama-index upgrades | Store a small version tag in the pickle; rebuild on mismatch (acceptable: re-ingest is user-visible but not destructive) |
| `st.session_state` loses the index between reruns if we forget to cache it | Load via `@st.cache_resource` keyed on chroma path |
| Scope selector invalidating chat engine resets conversation history | Document as intended behaviour (scope switch = fresh conversation); can be revisited if annoying |

## 11. Decisions (previously open)

1. **Scope switch resets chat history.** Switching scope is treated as starting a fresh conversation against a different corpus &mdash; no carry-over, no system message injection.
2. **Outline &amp; exercises are cached per scope in session state.** Switching back to a previously-generated scope shows the cached result without regenerating. Cache entries for a scope are cleared when any source in that scope is added or removed.
3. **No category rename UI for MVP.** Users who need to rename a category edit `sources.json` directly. Can be revisited later.
