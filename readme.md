# Vibe Study — AI Learning Assistant

Upload PDFs, YouTube links, or GitHub repos and chat with the content. Also generates a structured course outline and practice exercises via RAG (LlamaIndex + OpenAI).

## Features

- **Ingest**: PDF upload, YouTube transcript, GitHub repo (up to 30 files)
- **Chat**: scoped Q&A against your sources
- **Outline**: auto-generated structured outline per scope
- **Exercises**: practice questions generated from content

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| RAG | LlamaIndex 0.12 |
| Vector DB | ChromaDB |
| LLM / Embed | OpenAI (gpt-4o-mini / text-embedding-3-small) |
| PDF | pypdf |
| YouTube | youtube-transcript-api |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env → OPENAI_API_KEY=sk-...

# 3. Run
streamlit run app.py
```

## Running Tests

```bash
pytest
```

## Project Structure

```
app.py          # Streamlit entrypoint
ingestion.py    # PDF / YouTube / GitHub loaders
vector_store.py # ChromaDB index management
library.py      # Source registry (sources.json)
chat.py         # Chat engine
outline.py      # Outline generator
exercises.py    # Exercise generator
ui/             # Streamlit UI components
tests/          # pytest test suite
```
