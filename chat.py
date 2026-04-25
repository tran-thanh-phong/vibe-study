from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer


def create_chat_engine(index) -> CondensePlusContextChatEngine:
    retriever = index.as_retriever(similarity_top_k=5)
    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
    return CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        verbose=False,
    )


def chat(engine: CondensePlusContextChatEngine, user_message: str) -> str:
    response = engine.chat(user_message)
    return str(response)
