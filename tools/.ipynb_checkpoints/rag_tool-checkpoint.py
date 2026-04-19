from pydantic_ai import RunContext
from tools.dependencies import Deps

# Agent RAG tool
def search_local_dataset(ctx: RunContext[Deps], query: str) -> str:
    """
    Search the local dataset and return grounded results with source information.

    Use this when the answer may be present in the local corpus.
    Prefer this before web search when the question sounds like it could be answered
    from the local indexed dataset.
    """
    hits = ctx.deps.rag.search(query, limit=6)

    if not hits:
        return "No relevant results were found in the local dataset."

    parts = []
    for i, hit in enumerate(hits, start=1):
        title = hit.get("title", "") or "Untitled source"
        source_type = hit.get("source_type", "") or "unknown"
        source_path = hit.get("source_path", "") or "unknown"
        chunk_index = hit.get("chunk_index", -1)
        score = hit.get("score", 0.0)
        text = hit.get("text", "").strip()

        parts.append(
            f"""Result {i} 
            Title: {title}
            Source type: {source_type}
            Source path: {source_path}
            Chunk index: {chunk_index}
            Score: {score:.3f}
            Excerpt:{text}"""
        )

    return "\n\n---\n\n".join(parts)