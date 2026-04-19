from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIChatModel

from tools.dependencies import Deps, build_deps
from tools.web_search_tool import web_search_and_read
from tools.rag_tool import search_local_dataset

# Load necessary API keys
load_dotenv()

# Load agent instructions
BASE_DIR = Path(__file__).resolve().parent
INSTRUCTIONS = (BASE_DIR / "prompts" / "agent_instructions.md").read_text(encoding="utf-8")

def ensure_rag_ready(deps: Deps) -> None:
    """
    Ensure that the local RAG index exists before the agent runs.

    If the Qdrant collection for the local dataset is missing, this function
    builds the index from the files under the configured data root using the
    specified chunking parameters. If the index already exists, no rebuild is
    performed.
    """
    built = deps.rag.ensure_index(
        data_root="data",
        chunk_size=1200,
        overlap=200,
        batch_size=64,
    )
    if built:
        print("RAG index was missing and has now been built.")
    else:
        print("RAG index already exists.")

# Connection to MCP servers of Ansvar AI compliance tools
EU_law_total = MCPServerStreamableHTTP("https://mcp.ansvar.eu/eu-regulations/mcp")
US_law_total = MCPServerStreamableHTTP("https://mcp.ansvar.eu/us-regulations/mcp")
NL_law_total = MCPServerStreamableHTTP("https://mcp.ansvar.eu/law-nl/mcp")
Automotive_total = MCPServerStreamableHTTP("https://mcp.ansvar.eu/automotive/mcp")

# Avoid importing very computationally expensive tools
EU_ALLOWED = {
    "search_regulations",
    "get_definitions",
    "check_applicability",
    "compare_requirements",
    "get_article",  # expensive?
}
US_ALLOWED = {
    "search_regulations", 
    "check_applicability", 
    "compare_requirements",
    "map_controls", 
    "get_evidence_requirements", 
    "get_compliance_action_items", 
    #'get_section', # expensive?
}
NL_ALLOWED = {
    "search_legislation",
    "get_provision",
    "check_currency",
    "get_dutch_implementations",
    "validate_eu_compliance",
    "get_eu_basis",
    "get_provision_eu_basis",
}
CAR_ALLOWED = {
    "list_sources", 
    "search_requirements", 
    "list_work_products", 
}

# Selection of tools
EU_regulation_tools = EU_law_total.filtered(lambda ctx, tool_def: tool_def.name in EU_ALLOWED).prefixed("eu")
US_regulation_tools = US_law_total.filtered(lambda ctx, tool_def: tool_def.name in US_ALLOWED).prefixed("us")
NL_regulation_tools = NL_law_total.filtered(lambda ctx, tool_def: tool_def.name in NL_ALLOWED).prefixed("nl")
automotive_regulation_tools = Automotive_total.filtered(lambda ctx, tool_def: tool_def.name in CAR_ALLOWED).prefixed("automotive")

# Define EU AI Act compliance agent
model = OpenAIChatModel("gpt-4o-mini")
agent = Agent(
    model,
    deps_type=Deps,
    tools=[web_search_and_read, search_local_dataset],   # local Python tools for web-search and retrieval-augmented generation
    toolsets=[EU_regulation_tools, NL_regulation_tools, US_regulation_tools, automotive_regulation_tools],  # MCP-based toolsets
    instructions=INSTRUCTIONS, 
)

# Asynchronous implementation for notebook environments
async def run_agent(user_prompt: str):
    deps = build_deps()
    ensure_rag_ready(deps)
    try:
        async with agent:
            return await agent.run(user_prompt, deps=deps)
    finally:
        deps.rag.client.close()