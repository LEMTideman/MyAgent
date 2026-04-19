from __future__ import annotations

from pathlib import Path
from typing import Any

import cognee
from cognee.api.v1.search import SearchType
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver

BASE_DIR = Path(__file__).resolve().parent.parent
MEMORY_DATASET = "agent_memory"
AIRO_PATH = BASE_DIR / "ai_risk_ontology" / "airo.owl"


def build_ontology_config() -> Config | None:
    """
    Build the Cognee ontology config for AIRO.

    Returns None if the ontology file is not present, in which case Cognee
    will run without ontology grounding.
    """
    if not AIRO_PATH.exists():
        return None

    return {
        "ontology_config": {
            "ontology_resolver": RDFLibOntologyResolver(
                ontology_file=str(AIRO_PATH)
            )
        }
    }


def _to_text(results: Any) -> str:
    """
    Normalize Cognee results into a string for prompt injection.
    """
    if not results:
        return ""

    if isinstance(results, str):
        return results

    if isinstance(results, list):
        return "\n".join(str(item) for item in results)

    return str(results)


async def recall_memory(query: str, session_id: str, top_k: int = 5) -> str:
    """
    Retrieve both session memory and long-term memory relevant to the query.
    """
    # 1) session-aware lookup
    session_results = await cognee.recall(
        query_text=query,
        session_id=session_id,
        only_context=True,
        top_k=top_k,
    )

    # 2) explicit long-term graph lookup from the permanent dataset
    graph_results = await cognee.recall(
        query_text=query,
        datasets=[MEMORY_DATASET],
        query_type=SearchType.GRAPH_COMPLETION,
        only_context=True,
        top_k=top_k,
    )

    session_text = _to_text(session_results)
    graph_text = _to_text(graph_results)

    parts: list[str] = []
    if session_text:
        parts.append(f"Session memory:\n{session_text}")
    if graph_text:
        parts.append(f"Long-term memory:\n{graph_text}")

    return "\n\n".join(parts)


async def store_session_turn(user_prompt: str, assistant_text: str, session_id: str) -> None:
    """
    Store the latest turn in Cognee session memory.

    This is fast short-term memory, not the AIRO-grounded permanent graph.
    """
    text = (
        "Conversation turn\n"
        f"User: {user_prompt}\n"
        f"Assistant: {assistant_text}"
    )

    await cognee.remember(
        data=text,
        session_id=session_id,
        dataset_name=MEMORY_DATASET,
        self_improvement=False,
    )


async def store_long_term_memory(text: str) -> None:
    """
    Store durable memory in the permanent graph, grounded to AIRO when available.

    Use this for stable facts such as:
    - user preferences
    - project decisions
    - recurring constraints
    - AI system risk facts you want persisted across sessions
    """
    await cognee.add(
        data=text,
        dataset_name=MEMORY_DATASET,
        incremental_loading=True,
    )

    config = build_ontology_config()
    if config is None:
        await cognee.cognify(datasets=[MEMORY_DATASET])
    else:
        await cognee.cognify(
            datasets=[MEMORY_DATASET],
            config=config,
        )


async def remember_fact(text: str) -> str:
    """
    Tool-friendly wrapper for saving durable long-term memory.
    """
    await store_long_term_memory(text)
    return "Saved to long-term memory."