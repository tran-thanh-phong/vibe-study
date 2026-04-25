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
