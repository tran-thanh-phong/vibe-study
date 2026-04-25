# AI Learning Assistant — Design Spec

**Date:** 2026-04-24
**Scope:** MVP final course project (3-day build)
**Status:** Approved

---

## 1. Overview

A local Streamlit app that ingests learning materials (PDF, YouTube, GitHub) and provides:
- RAG-based chat with conversation history
- Auto-generated course outline
- Auto-generated practice exercises (easy / medium / hard)

No authentication, no database persistence beyond the local vector store, runs entirely on localhost.

---

## 2. Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| UI | Streamlit | Rapid demo UI, no frontend build step |
| AI Framework | LlamaIndex | Purpose-built for RAG pipelines, minimal boilerplate |
| LLM | gpt-4o-mini | Fast, cheap, sufficient quality for demo |
| Embeddings | text-embedding-3-small | Cheap, fast, good quality |
| Vector DB | Chroma (local) | No external service, persists to disk |
| PDF extraction | pypdf | Lightweight, no external service |
| YouTube transcript | youtube-transcript-api | No API key, pulls existing captions |
| GitHub ingestion | GitHub REST API | No git clone required, works for public repos |

---

## 3. Project Structure

```
vibe-study/
├── app.py                  # Streamlit UI only
├── ingestion.py            # PDF + YouTube + GitHub extraction & chunking
├── chat.py                 # RAG query engine with conversation history
├── outline.py              # Outline generation from full corpus
├── exercises.py            # Exercise generation (easy/medium/hard)
├── vector_store.py         # Chroma setup, index creation/loading
├── requirements.txt
└── .env                    # OPENAI_API_KEY
```

---

## 4. Ingestion Pipeline (`ingestion.py`)

Accepts any combination of PDF file, YouTube URL, GitHub URL. Returns a merged list of LlamaIndex `Document` objects.

### PDF
- Library: `pypdf`
- Each page → one `Document`

### YouTube
- Library: `youtube-transcript-api`
- Parse video ID from URL
- Full transcript → one `Document`

### GitHub
- GitHub REST API: `GET /repos/{owner}/{repo}/git/trees/{branch}?recursive=1`
- Always includes `README.md`
- Filters by extension: `.py`, `.js`, `.ts`, `.md`, `.java`, `.go`
- Each file → one `Document` with filename as metadata
- Public repos only

### Merging
- All documents collected into single list
- Passed to `vector_store.py` for chunking + embedding
- Chunk size: 512 tokens, overlap: 50 tokens

---

## 5. Embedding & Vector Store (`vector_store.py`)

- Index type: `VectorStoreIndex` (LlamaIndex)
- Backend: Chroma in local persistent mode (`./chroma_db/`)
- On each "Process Content" click: drops and recreates the Chroma collection to avoid polluting retrieval with stale vectors from a previous run
- Stored in `st.session_state.index` after build

---

## 6. Chat — RAG + Conversation History (`chat.py`)

- Engine: `CondensePlusContextChatEngine` (LlamaIndex)
- Condenses chat history + new question → standalone retrieval query
- Retrieves top-5 chunks from vector index
- Generates answer with `gpt-4o-mini` using retrieved context + full history
- Engine instance stored in `st.session_state.chat_engine`
- Chat messages stored in `st.session_state.messages` (list of `{role, content}`)
- "Clear chat" button resets engine + messages without re-processing documents

---

## 7. Outline Generation (`outline.py`)

- Index type: `SummaryIndex` (LlamaIndex) — uses full corpus, not top-k retrieval
- One-shot generation triggered by button click
- Prompt:
  ```
  Generate a structured outline from this content.
  Format: numbered sections, each with a 1-2 sentence description.
  ```
- Output: markdown string stored in `st.session_state.outline`
- Rendered via `st.markdown()`

---

## 8. Exercise Generation (`exercises.py`)

- Same `SummaryIndex` as outline generation
- One-shot generation triggered by button click
- Prompt:
  ```
  Generate 3 practice exercises based on this content:
  - 1 easy (recall/definition)
  - 1 medium (application)
  - 1 hard (analysis/synthesis)
  For each: state the exercise clearly and provide a sample answer.
  ```
- Output: markdown string stored in `st.session_state.exercises`
- Rendered via `st.markdown()`

---

## 9. Streamlit UI Layout (`app.py`)

### Sidebar
- PDF file uploader (`st.file_uploader`)
- YouTube URL text input
- GitHub repo URL text input
- "Process Content" button with spinner
- Status indicator: shows "Ready ✓" once index is built

### Main Area — 3 Tabs
| Tab | Content |
|-----|---------|
| 💬 Chat | Chat bubble UI, text input at bottom, "Clear chat" button |
| 📋 Outline | "Generate Outline" button, markdown output |
| 🏋️ Exercises | "Generate Exercises" button, markdown output |

### UX Rules
- All tabs disabled until "Process Content" completes successfully
- Each generation button shows a spinner while running
- Results cached in `session_state` — no regeneration on Streamlit rerenders
- At least one source (PDF, YouTube, or GitHub) required before processing

---

## 10. Session State Map

| Key | Type | Set by |
|-----|------|--------|
| `index` | `VectorStoreIndex` | `vector_store.py` after ingestion |
| `chat_engine` | `CondensePlusContextChatEngine` | `chat.py` on first message |
| `messages` | `list[dict]` | `app.py` chat tab |
| `outline` | `str` | `outline.py` |
| `exercises` | `str` | `exercises.py` |
| `ready` | `bool` | `app.py` after processing completes |

---

## 11. Error Handling

| Scenario | Handling |
|----------|----------|
| YouTube video has no captions | Show warning, skip YouTube source |
| GitHub repo is private or not found | Show error message, skip GitHub source |
| PDF is scanned (no text layer) | Show warning, skip PDF source |
| No sources provided | Disable "Process Content" button |
| OpenAI API error | Show `st.error()` with message |

---

## 12. Dependencies (`requirements.txt`)

```
llama-index
llama-index-vector-stores-chroma
llama-index-embeddings-openai
llama-index-llms-openai
chromadb
pypdf
youtube-transcript-api
streamlit
python-dotenv
requests
```

---

## 13. Success Criteria

- User can upload any combination of PDF, YouTube URL, GitHub URL
- Chat returns relevant, contextually accurate answers
- Follow-up questions correctly reference prior conversation turns
- Outline is logically structured with clear sections
- Exercises are meaningful and cover easy/medium/hard difficulty
- End-to-end demo runs without errors on localhost

---

## 14. Out of Scope

- User authentication
- Multi-session / multi-course management
- Progress tracking
- Private GitHub repositories
- YouTube videos without existing captions
- Mobile-responsive UI
