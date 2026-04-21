from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel


class ScopeDecision(BaseModel):
    """
    Structured classification result returned by the scope guard.

    This model records whether a user prompt is within the agent's allowed topic boundary, 
    assigns it to a coarse category, and provides a short explanation for the decision.
    """
    allowed: bool = Field(
        description="True only if the user's primary intent is about AI regulation/compliance/governance."
    )
    category: Literal["ai_regulation", "out_of_scope"]
    reason: str = Field(
        description="Very short reason for the decision.",
        max_length=200,
    )


SCOPE_BLOCK_MESSAGE = (
    "I only answer questions about AI regulation, AI governance, AI compliance, "
    "AI standards used for compliance, and closely related legal or assurance topics."
)

SCOPE_INSTRUCTIONS = """
You are a strict scope classifier for an AI regulation agent.

Allow ONLY prompts whose primary intent is about:
- AI regulation or AI law
- AI governance, AI compliance, or AI assurance
- standards used in AI compliance or assurance
- conformity assessment, audits, evidence artifacts, enforcement, obligations
- sector-specific rules, safety, or cybersecurity requirements WHEN asked in relation to an AI system

Geography is unrestricted.

Examples that should be ALLOWED:
- EU AI Act questions
- US, UK, Canada, China, OECD, UNESCO, or other AI regulation/governance questions
- ISO/IEC 42001, ISO/IEC 23894, NIST AI RMF, harmonized standards, evidence artifacts
- questions about whether an AI system is high-risk
- questions about compliance mapping between AI rules and standards

Examples that should be REJECTED:
- general Python/programming help
- general machine learning engineering
- general legal research unrelated to AI
- general cybersecurity unrelated to AI systems
- travel, sports, cooking, weather, etc.

Borderline rule:
If the prompt mixes topics, allow it only if AI regulation/compliance/governance is the main topic.

Return structured output only.
""".strip()


# Keep the guard lightweight and independent from app dependencies.
_scope_model = OpenAIChatModel("gpt-4o-mini")

scope_agent = Agent(
    _scope_model,
    output_type=ScopeDecision,
    instructions=SCOPE_INSTRUCTIONS,
)


async def check_prompt_scope(user_prompt: str) -> ScopeDecision:
    """
    Classify a user prompt as in-scope or out-of-scope for the agent.

    Args:
        user_prompt: The raw user question to evaluate.

    Returns:
        A ScopeDecision object containing the classifier's decision and
        a brief reason for it.
    """
    result = await scope_agent.run(user_prompt)
    return result.output


async def block_if_out_of_scope(user_prompt: str) -> str | None:
    """
    Check whether a user prompt is allowed and return a block message if not.

    Args:
        user_prompt: The raw user question to evaluate.

    Returns:
        The standard refusal message if the prompt is out of scope, or None
        if the prompt is allowed to proceed to the main agent.
    """
    decision = await check_prompt_scope(user_prompt)
    if decision.allowed:
        return None
    return SCOPE_BLOCK_MESSAGE