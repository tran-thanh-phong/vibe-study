from unittest.mock import MagicMock, patch


class TestCreateChatEngine:
    def test_scope_all_passes_no_filter(self):
        from chat import create_chat_engine
        index = MagicMock()
        index.as_retriever = MagicMock()
        create_chat_engine(index, scope="All")
        kwargs = index.as_retriever.call_args.kwargs
        assert kwargs.get("filters") is None
        assert kwargs["similarity_top_k"] == 5

    def test_scope_category_adds_metadata_filter(self):
        from chat import create_chat_engine
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
        index = MagicMock()
        index.as_retriever = MagicMock()
        create_chat_engine(index, scope="Day 1")
        filters = index.as_retriever.call_args.kwargs["filters"]
        assert isinstance(filters, MetadataFilters)
        assert len(filters.filters) == 1
        f = filters.filters[0]
        assert f.key == "category"
        assert f.value == "Day 1"
