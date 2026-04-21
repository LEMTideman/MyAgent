# Input guarderails
# Before running our agent, we verify that the user prompt is in scope using another agent.
# Use < uv run pytest -s -v tests/test_scope_agent.py > in Powershell to run evaluation. 

import pytest

from guards.scope_guard import check_prompt_scope, block_if_out_of_scope


@pytest.mark.asyncio
async def test_scope_guard_allows_ai_regulation_question():
    decision = await check_prompt_scope(
        "Which obligations apply to providers of general purpose AI models under the EU AI Act?"
    )
    assert decision.allowed is True


@pytest.mark.asyncio
async def test_scope_guard_blocks_general_programming_question():
    blocked = await block_if_out_of_scope(
        "Write me a Python FastAPI app."
    )
    assert blocked is not None


@pytest.mark.asyncio
async def test_scope_guard_allows_ai_standard_question():
    decision = await check_prompt_scope(
        "How does ISO/IEC 42001 support AI governance and compliance?"
    )
    assert decision.allowed is True


@pytest.mark.asyncio
async def test_scope_guard_blocks_irrelevant_question():
    blocked = await block_if_out_of_scope("What are the best vegetarian restaurants of Paris?")
    assert blocked is not None


@pytest.mark.asyncio
async def test_scope_guard_allows_ai_regulation_question():
    decision = await check_prompt_scope("How is generative AI regulated in Singapore? What are the pros and cons of the Singaporean approach versus the European and Chineses approaches?")
    assert decision.allowed is True