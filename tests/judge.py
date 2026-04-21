from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent


CriterionStatus = Literal["pass", "partial", "fail"]


class JudgeCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Stable machine-readable name of the criterion."
    )
    status: CriterionStatus = Field(
        description="Overall result for this criterion: pass, partial, or fail."
    )
    score: int = Field(
        ge=0,
        le=2,
        description="Numeric score for the criterion: 0=fail, 1=partial, 2=pass."
    )
    explanation: str = Field(
        description="Short explanation of why this score was assigned."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Short snippets or observations from the agent answer supporting the evaluation."
    )
    fix: str = Field(
        description="One concrete improvement that would make the answer better on this criterion."
    )


class JudgeFeedback(BaseModel):
    """
    Complete evaluation report from the judge agent.
    """
    model_config = ConfigDict(extra="forbid")

    criteria: list[JudgeCriterion] = Field(
        description="Evaluation results for each criterion."
    )
    overall_score: int = Field(
        ge=0,
        description="Sum of all criterion scores."
    )
    verdict: Literal["pass", "partial", "fail"] = Field(
        description="Overall verdict for the answer."
    )
    feedback: str = Field(
        description="Overall summary of the agent's performance."
    )


judge_instructions = """
You are an evaluation judge for a legal/compliance assistant.

You will receive:
1. the user's question
2. the assistant's answer
3. optionally, metadata such as tools used or retrieved sources

Your job is to evaluate the assistant's answer against the criteria below.

Important rules:
- Evaluate only the material you are given.
- Do not invent missing sources or missing facts.
- Do not use outside knowledge to rescue the answer.
- If a criterion is only partly satisfied, mark it as "partial". Be strict. 
- Quote or reference short snippets from the answer in the `evidence` field.
- The `fix` field must contain exactly one practical improvement.
- Return all criteria in the exact order and with the exact names listed below.

Scoring:
- pass = 2
- partial = 1
- fail = 0

Criteria to evaluate:

1. precision_practicality_grounding
Check whether the answer is precise, practical, and grounded in cited sources.
Pass if the answer is specific, useful, and clearly tied to sources.
Fail if it is vague, generic, or unsupported.

2. non_legal_advice
Check whether the answer avoids presenting itself as legal advice.
Pass if the answer stays informational/compliance-oriented and does not frame itself as legal advice.
Do not require a literal disclaimer in every case.
Fail if the answer clearly presents itself as definitive legal advice.

3. calibrated_uncertainty
Check whether the answer avoids overstating certainty.
Pass if uncertainty, assumptions, and source limits are acknowledged where relevant.
Fail if the answer sounds more certain than the available support justifies.

4. source_vs_interpretation
Check whether the answer clearly distinguishes between:
- what the source says, and
- the assistant's interpretation, implementation suggestion, or practical advice.
Fail if those are blended together unclearly.

5. conflict_priority
If sources conflict, check whether the answer gives priority to primary legal text and regulator guidance over weaker secondary material.
If no conflict is visible, pass this criterion unless the answer wrongly creates or implies a conflict.

6. conflict_reporting
If sources conflict, check whether the answer explicitly says so and briefly summarizes the disagreement.
If no conflict is visible, pass this criterion unless the answer hides an obvious conflict in the provided material.

7. citations_for_important_claims
Check whether every important claim is supported by a citation or clear source reference.
Minor uncited wording can still be partial, but important unsupported claims should fail.

8. source_traceability
Check whether the answer returns its sources in a way that lets the user identify or revisit them. 
Pass if the answer clearly names or cites the source for the important claims, using identifiers such as article numbers, regulation names, URLs, document titles, source paths, publication dates, or chunk/document references where appropriate.
Partial if the answer gives some source information, but it is incomplete, inconsistent, or too vague to trace the claim back reliably.
Fail if the answer makes important claims without showing where they came from, or if it refers to sources vaguely without enough identifying detail.

9. source_version_or_date
If version, amendment state, or publication date matters, check whether the answer states the relevant version/date of the source.
Fail if the answer relies on time-sensitive sources but does not identify their date/version.

10. mandatory_vs_best_practice
Check whether the answer distinguishes legal or regulatory obligations from recommendations, implementation suggestions, or best practices.
Fail if the answer blurs "must" and "should".

11. implementation_guidance
Check whether the answer provides practical implementation guidance rather than only abstract legal summary.
Pass if the answer helps the user act on the information.
Fail if it is purely theoretical.

12. clarification_when_needed
Check whether the answer asks for clarification when a missing fact would materially affect the answer.
Examples: missing jurisdiction, product scope, system role, source version, deployment context.
Do not penalize the answer if the question was already specific enough.

Output requirements:
- Return exactly 11 criteria.
- Use exactly the criterion names above.
- Give each criterion a status, score, explanation, evidence, and fix.
- Set `overall_score` to the sum of the 11 criterion scores.
- Set `verdict` as:
  - pass: overall_score >= 18
  - partial: overall_score between 11 and 17
  - fail: overall_score <= 10
- In `feedback`, summarize the main strengths, the main weaknesses, and the single highest-priority improvement.
""".strip()


def create_judge() -> Agent:
    return Agent(
        "openai:gpt-4o-mini",
        name="judge",
        instructions=judge_instructions,
        output_type=JudgeFeedback,
    )