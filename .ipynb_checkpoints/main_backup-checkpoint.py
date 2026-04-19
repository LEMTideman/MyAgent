from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIChatModel

from tools.dependencies import Deps, build_deps
from tools.web_search_tool import web_search_and_read

# Load necessary API keys
load_dotenv()

# Load agent instructions
BASE_DIR = Path(__file__).resolve().parent
INSTRUCTIONS = (BASE_DIR / "prompts" / "agent_instructions.md").read_text(encoding="utf-8")

# Connection to MCP servers
EU_law_total = MCPServerStreamableHTTP("https://mcp.ansvar.eu/eu-regulations/mcp")
US_law_total = MCPServerStreamableHTTP("https://mcp.ansvar.eu/us-regulations/mcp")
NL_law_total = MCPServerStreamableHTTP("https://mcp.ansvar.eu/law-nl/mcp")
Automotive_total = MCPServerStreamableHTTP("https://mcp.ansvar.eu/automotive/mcp")

#EU_law_total = MCPServerStreamableHTTP("https://eu-regulations-mcp.vercel.app/mcp")
#US_law_total = MCPServerStreamableHTTP("https://us-regulations-mcp.vercel.app/mcp")
#NL_law_total = MCPServerStreamableHTTP("https://dutch-law-mcp.vercel.app/mcp")
#Automotive_total = MCPServerStreamableHTTP("https://automotive-cybersecurity-mcp.vercel.app/mcp")

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
    tools=[web_search_and_read],   # local Python tools
    toolsets=[EU_regulation_tools, NL_regulation_tools, US_regulation_tools, automotive_regulation_tools],  # MCP-based toolsets
    instructions=(
        "Use MCP tools when they are relevant. Mention that you are using the Ansvar AI compiance tools."
        "Use web_search_and_read when you need broader or fresher web information."
        "When using web sources, cite the URLs you relied on."
        "Use search_local_dataset when the question may be answered from the local dataset (YouTube transcripts and Docling-processed PDFs)."
        "When you use local retrieval results, cite the LOCAL markers such as [LOCAL-1]."
    ),
)

# Asynchronous implementation for notebook environments
async def run_agent(user_prompt: str):
    async with agent:
        return await agent.run(user_prompt, deps=build_deps())