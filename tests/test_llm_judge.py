# LLM-as-a-judge evaluation approach
# We run the agent on test prompts and ask another LLM to assess the quality of the agent's answer against criteria 
# such as correctness and relevance. It is an LLM-based assessment of the agent. 

import pytest

from main import run_agent
from tests.judge import create_judge

@pytest.mark.asyncio
async def test_agent_with_llm_judge():
    question = "How is AI regulated in Singapore? What are the pros and cons of the Singaporean approach versus the European and Chineses approaches?"
    result = await run_agent(question)

    judge = create_judge()
    judgment = await judge.run(
        f"""
        USER QUESTION:
        {question}

        AGENT ANSWER:
        {result.output}
        """
    )

    feedback = judgment.output

    assert feedback.verdict in {"pass", "partial"}