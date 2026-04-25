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
