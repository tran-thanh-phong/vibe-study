from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter


def create_chat_engine(index, scope: str = "All") -> CondensePlusContextChatEngine:
    kwargs = {"similarity_top_k": 5, "filters": None}
    if scope != "All":
        kwargs["filters"] = MetadataFilters(
            filters=[MetadataFilter(key="category", value=scope)]
        )
    retriever = index.as_retriever(**kwargs)
    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
    return CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        verbose=False,
    )


def chat(engine: CondensePlusContextChatEngine, user_message: str) -> str:
    response = engine.chat(user_message)
    return str(response)
