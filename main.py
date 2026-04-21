from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIChatModel

from guards.scope_guard import block_if_out_of_scope
from tools.dependencies import Deps, build_deps
from tools.web_search_tool import web_search_and_read
from tools.rag_tool import search_local_dataset

# Load necessary API keys
load_dotenv()

# Load agent instructions
BASE_DIR = Path(__file__).resolve().parent
INSTRUCTIONS = (BASE_DIR / "prompts" / "agent_instructions.md").read_text(encoding="utf-8")

# Connection to MCP servers of Ansvar AI compliance tools
EU_law_total = MCPServerStreamableHTTP("https://mcp.ansvar.eu/eu-regulations/mcp")
US_law_total = MCPServerStreamableHTTP("https://mcp.ansvar.eu/us-regulations/mcp")
NL_law_total = MCPServerStreamableHTTP("https://mcp.ansvar.eu/law-nl/mcp")
Automotive_total = MCPServerStreamableHTTP("https://mcp.ansvar.eu/automotive/mcp")

# Choose MCP-based tools from Ansvar-Systems
# Avoid importing very computationally expensive tools
EU_ALLOWED = {
    "search_regulations",
    "get_definitions",
    "check_applicability",
    "compare_requirements",
    "get_article",
}
US_ALLOWED = {
    "search_regulations", 
    "check_applicability", 
    "compare_requirements",
    "map_controls", 
    "get_evidence_requirements", 
    "get_compliance_action_items", 
    "get_section", 
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

# Define AI regulation compliance agent
model = OpenAIChatModel("gpt-4o-mini")
agent = Agent(
    model,
    deps_type=Deps,
    tools=[web_search_and_read, search_local_dataset],  # local Python tools for web-search and retrieval-augmented generation
    toolsets=[EU_regulation_tools, NL_regulation_tools, US_regulation_tools, automotive_regulation_tools],  # MCP-based toolsets
    instructions=INSTRUCTIONS, 
)

async def run_agent(user_prompt: str):
    # Check whether the user prompt is within the allowed scope, and return a refusal message if it is not
    # You want to block out-of-scope prompts before incurring costs
    blocked_message = await block_if_out_of_scope(user_prompt)
    if blocked_message is not None:
        return blocked_message

    # Create the shared runtime dependencies the agent needs, 
    # such as the web-search clients and the local RAG system.
    deps = build_deps()

    # Run the agent with those dependencies and then closes the RAG and web-search connections
    # Asynchronous implementation for notebook environments
    try:
        async with agent:
            return await agent.run(user_prompt, deps=deps)
    finally:
        deps.rag.client.close()
        deps.brave.session.close()