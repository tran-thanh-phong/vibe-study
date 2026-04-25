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
