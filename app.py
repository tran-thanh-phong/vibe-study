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
