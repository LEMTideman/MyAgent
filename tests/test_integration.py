# Integration test
# We verify the agent’s routing behavior using a series of tests. 
# We run the agent end-to-end with realistic prompts in order to check whether the correct tools are used. 
# Use < uv run pytest -s -v tests/test_integration.py > in Powershell to run tests. 

from __future__ import annotations

import os
import sys
import functools
from pathlib import Path
from dataclasses import dataclass

import httpx
import pytest
from pydantic_ai import Agent

import main
from tools.dependencies import build_deps
from tools.rag_tool import search_local_dataset
from tools.web_search_tool import web_search_and_read


@dataclass(frozen=True)
class RoutingCase:
    name: str
    prompt: str
    expected_routes: tuple[str, ...]


@dataclass
class TraceHarness:
    calls: list[str]
    agent: Agent


CASES = [
    RoutingCase(
        name="rag",
        prompt=(
            "In order to help agentic AI providers comply with the EU AI Act, Luca Nannini et al. propose a practical"
            "compliance architecture of twelve sequential steps. Explain compliance sequence to me."
        ),
        expected_routes=("rag_tool",),
    ),
    RoutingCase(
        name="automotive_mcp",
        prompt=(
            "We are preparing compliance evidence for a vehicle type approval submission."
            "What does UNECE R155 Article 7 require for a CSMS, and which ISO/SAE 21434 clauses and work products map to it?"
        ),
        expected_routes=("mcp:automotive",),
    ),
    RoutingCase(
        name="nl_mcp",
        prompt=(
            "We are checking a court filing for accurate Dutch statutory references. Validate the citation 'Wetboek van Strafrecht artikel 138b',"
            "tell me whether it is still in force, and quote the current provision text in proper Dutch legal citation format."
        ),
        expected_routes=("mcp:nl",),
    ),
    RoutingCase(
        name="websearch",
        prompt=(
            "China's new law on AI anthropomorphism was officially enacted in April 2026. What measures does it propose to protect children?"
        ),
        expected_routes=("web_tool",),
    ),
    RoutingCase(
        name="websearch",
        prompt=(
            "What is the latest status of the EU Digital Omnibus on AI, and which parts of the AI Act would it change if adopted?"
            "What difference will it make for small enterprises?"
        ),
        expected_routes=("web_tool",),
    ),
    RoutingCase(
        name="eu_mcp_automotive_mcp",
        prompt=(
            "For an AI-based driver monitoring system in a connected passenger car sold in the EU, which exact EU AI Act provisions determine whether it is high-risk,"
            "and which UNECE R155/R156 and ISO/SAE 21434 requirements govern its cybersecurity risk assessment, software update process, and required evidence artifacts?"
        ),
        expected_routes=("mcp:eu", "mcp:automotive"),
    ),
]


REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "BRAVE_SEARCH_API_KEY",
    "JINA_API_KEY",
]


@pytest.fixture(scope="session", autouse=True)
def check_env_vars() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        pytest.skip(
            "Missing required environment variables: " + ", ".join(missing),
            allow_module_level=True,
        )


@pytest.fixture()
def trace_harness(monkeypatch: pytest.MonkeyPatch) -> TraceHarness:
    calls: list[str] = []

    @functools.wraps(search_local_dataset)
    def traced_rag_tool(ctx, query: str) -> str:
        calls.append("rag_tool")
        return search_local_dataset(ctx, query)

    @functools.wraps(web_search_and_read)
    def traced_web_tool(
        ctx,
        query: str,
        num_results: int = 5,
        max_chars_per_doc: int = 20_000,
        use_readerlm_v2: bool = True,
    ):
        calls.append("web_tool")
        return web_search_and_read(
            ctx,
            query=query,
            num_results=num_results,
            max_chars_per_doc=max_chars_per_doc,
            use_readerlm_v2=use_readerlm_v2,
        )

    original_send = httpx.AsyncClient.send

    async def traced_send(self, request, *args, **kwargs):
        url = str(request.url)

        try:
            body = request.content.decode("utf-8", errors="ignore")
        except Exception:
            body = ""

        is_actual_tool_call = '"method":"tools/call"' in body or '"method": "tools/call"' in body

        if is_actual_tool_call:
            if "mcp.ansvar.eu/automotive/" in url:
                calls.append("mcp:automotive")
            elif "mcp.ansvar.eu/law-nl/" in url:
                calls.append("mcp:nl")
            elif "mcp.ansvar.eu/eu-regulations/" in url:
                calls.append("mcp:eu")
            elif "mcp.ansvar.eu/us-regulations/" in url:
                    calls.append("mcp:us")

        return await original_send(self, request, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", traced_send)

    test_agent = Agent(
        main.model,
        deps_type=main.Deps,
        tools=[traced_web_tool, traced_rag_tool],
        toolsets=[
            main.EU_regulation_tools,
            main.NL_regulation_tools,
            main.US_regulation_tools,
            main.automotive_regulation_tools,
        ],
        instructions=main.INSTRUCTIONS,
    )

    return TraceHarness(calls=calls, agent=test_agent)


def unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


async def run_case(agent: Agent, prompt: str) -> str:
    deps = build_deps()
    try:
        async with agent:
            result = await agent.run(prompt, deps=deps)
        return str(result.output)
    finally:
        deps.rag.client.close()
        deps.brave.session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
async def test_agent_routing(case: RoutingCase, trace_harness: TraceHarness) -> None:
    answer = await run_case(trace_harness.agent, case.prompt)
    observed = unique_in_order(trace_harness.calls)

    print(f"\nCASE: {case.name}")
    print(f"PROMPT: {case.prompt}")
    print(f"ROUTES USED: {observed}")
    print("ANSWER:")
    print(answer)

    assert answer.strip(), "The agent returned an empty answer."

    for expected_route in case.expected_routes:
        assert expected_route in observed, (
            f"Expected route '{expected_route}' was not observed. "
            f"Observed routes: {observed}"
        )
