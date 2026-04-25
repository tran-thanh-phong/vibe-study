from llama_index.core import SummaryIndex

_EXERCISE_PROMPT = (
    "Generate 3 practice exercises based on the content provided.\n\n"
    "**Easy (Recall/Definition)**\n"
    "Exercise: [state the exercise]\n"
    "Sample Answer: [provide a sample answer]\n\n"
    "**Medium (Application)**\n"
    "Exercise: [state the exercise]\n"
    "Sample Answer: [provide a sample answer]\n\n"
    "**Hard (Analysis/Synthesis)**\n"
    "Exercise: [state the exercise]\n"
    "Sample Answer: [provide a sample answer]"
)


def generate_exercises(documents) -> str:
    index = SummaryIndex.from_documents(documents)
    query_engine = index.as_query_engine(response_mode="tree_summarize")
    response = query_engine.query(_EXERCISE_PROMPT)
    return str(response)
