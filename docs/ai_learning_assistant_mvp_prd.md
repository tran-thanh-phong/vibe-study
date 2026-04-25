# AI Learning Assistant (Course Chat + Outline Generator)

## 1. MVP Context Brief

**Goal**: Build a simple AI-powered tool that allows users to upload learning materials (PDF + YouTube + Github) and interact via chat, while automatically generating a structured outline and practice exercises.

**Scope**: Final course project (short-term). Focus on demonstrating end-to-end AI value, not production readiness.

**Core Value Proposition**:

- Turn raw learning materials into structured knowledge
- Enable contextual Q&A via AI
- Generate actionable learning outputs (outline + exercises)

---

## 2. Target User

- Learners taking online courses
- Developers / students reviewing technical materials

---

## 3. User Journey Map

1. User uploads:

   - PDF (slide)
   - YouTube link
   - Github link

2. System processes content:

   - Extract text (PDF)
   - Extract transcript (YouTube)
   - Extract Github codebase

3. User interacts:

   - Ask questions via chat

4. System generates:

   - Course outline
   - Practice exercises

---

## 4. Feature Backlog

### Core Features (MVP)

1. Content Ingestion

   - Upload PDF
   - Input YouTube URL
   - Input Github URL

2. Content Processing

   - Extract text
   - Chunk content
   - Generate embeddings

3. Chat (RAG)

   - Ask questions
   - Retrieve relevant chunks
   - Generate answer using LLM

4. Outline Generator

   - Generate structured outline from content

5. Exercise Generator

   - Generate 3 exercises:
     - Easy
     - Medium
     - Hard

---

## 5. MVP Feature List (Final Scope)

### Must-have

- Upload PDF
- Input YouTube link
- Input GitHub repository link
- Chat with content (RAG)
- Generate outline
- Generate exercises

### Nice-to-have (if time permits)

- Regenerate button
- Copy/export result (if time permits)

- Regenerate button
- Copy/export result

### Out of Scope

- User authentication
- Multi-course management
- Progress tracking
- Database persistence
- Advanced UI/UX

---

## 6. Functional Requirements

### 6.1 Upload Module

- Accept PDF file
- Accept YouTube URL
- Accept GitHub repository URL

### 6.2 Processing Module

- Extract text from PDF
- Extract transcript from YouTube
- Extract GitHub repository content:
  - README.md
  - Selected source files (e.g., .py, .js)
  - Code comments
- Merge into single corpus
- Split into chunks (~500 tokens)

- Extract text from PDF
- Extract transcript from YouTube
- Merge into single corpus
- Split into chunks (\~500 tokens)

### 6.3 Embedding & Storage

- Generate embeddings
- Store in vector database (Chroma)

### 6.4 Chat Module

- Input: user question
- Retrieve top-k relevant chunks
- Generate response using LLM

### 6.5 Outline Generation

- Input: full content
- Output:
  - Structured outline
  - Sections + short descriptions

### 6.6 Exercise Generation

- Input: full content
- Output:
  - 3 exercises (easy, medium, hard)

---

## 7. Non-Functional Requirements

- Response time: < 5 seconds (acceptable for demo)
- System runs locally
- No authentication required

---

## 8. Technical Architecture

### Stack

- Backend: Python
- Framework: LangChain or LlamaIndex
- Vector DB: Chroma (local)
- LLM: OpenAI API
- UI: Streamlit

### High-level Flow

Input → Processing → Embedding → Vector DB → Retrieval → LLM → Output

---

## 9. Prompt Design

### Outline Prompt

```
Generate a structured outline from the following content:
- Divide into main sections
- Each section has a short description
```

### Exercise Prompt

```
Based on the content, generate 3 exercises:
- 1 easy
- 1 medium
- 1 hard
```

---

## 10. Success Criteria

- User can upload materials successfully
- Chat returns relevant answers
- Outline is logically structured
- Exercises are meaningful and actionable
- End-to-end demo works smoothly

---

## 11. Demo Plan

1. Upload PDF + YouTube + GitHub repo
2. Show processing (including code ingestion)
3. Ask question via chat (including code-related question)
4. Generate outline
5. Generate exercises

---

## 12. Future Enhancements Future Enhancements

- Multi-session learning
- Progress tracking
- Personal notes integration
- Recommendation engine
- Multi-course management

---

## 13. Risks & Mitigation

| Risk                    | Mitigation                 |
| ----------------------- | -------------------------- |
| Poor transcript quality | Use fallback or clean text |
| Irrelevant answers      | Tune chunk size & top-k    |
| Slow response           | Reduce context size        |

---

## 14. Timeline

- Day 1: Ingestion + Embedding + RAG
- Day 2: UI + Outline + Exercises
- Day 3: Testing + Demo Prep

---

## 15. Summary

A focused MVP that demonstrates:

- AI-powered knowledge retrieval
- Automated content structuring
- Learning enhancement via generated exercises

Designed for fast implementation and strong demo impact.

