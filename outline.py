from llama_index.core import SummaryIndex

_OUTLINE_PROMPT = (
    "Generate a structured outline from the content provided.\n"
    "Format your response as numbered sections:\n"
    "1. [Section Title]\n"
    "   Description: [1-2 sentence description]\n\n"
    "2. [Section Title]\n"
    "   Description: [1-2 sentence description]\n\n"
    "Cover all major topics in the content."
)


def generate_outline(documents) -> str:
    index = SummaryIndex.from_documents(documents)
    query_engine = index.as_query_engine(response_mode="tree_summarize")
    response = query_engine.query(_OUTLINE_PROMPT)
    return str(response)
