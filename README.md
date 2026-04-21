# AI Regulation Research Assistant

A domain-specific research assistant for questions about AI regulation, AI governance, AI compliance developed in the context of the [AI engineering course by Alexey Grigorev](https://maven.com/alexey-grigorev/from-rag-to-agents). 

Our agent is designed to help with practical research questions such as:
- which legal obligations apply to a given AI system,
- how regulatory requirements map to compliance controls or evidence artifacts,
- how different jurisdictions approach AI regulation,
- and how sector-specific requirements interact with AI governance.

It is especially useful for questions about EU and US AI regulation, while also supporting related Dutch legal research and automotive compliance use cases.

---

## Repo overview

Our agent combines:
- a strict scope guard that blocks irrelevant prompts,
- a local RAG system for grounded retrieval from a curated local dataset,
- a web search + webpage reading pipeline for current or external material,
- multiple MCP-based specialist toolsets for selected legal and compliance domains,
- integration tests that verify whether the agent routes questions to the correct tools,
- LLM-as-a-judge evaluation that scores answer quality against structured legal/compliance criteria.


---

## Tooling

### 1. Local RAG tool
The agent includes a retrieval tool that searches a local indexed dataset before falling back to web search where appropriate.

What it does:
- searches the local corpus semantically,
- returns the top matches,
- and includes source information such as title, source type, source path, chunk index.

This is useful for:
- internal knowledge bases,
- curated research corpora,
- reusable background material on AI governance and compliance.

The local RAG system is initialized automatically when dependencies are built. The local database includes YouTube video transcripts and PDF documents on AI regulation from multiple sources (e.g. arXiv, Substack). Data is collected from publicly available internet sources and used solely for educational purposes.

---

### 2. Web search + webpage reading tool
The web tool combines:
- Brave Search for finding relevant pages,
- Jina Reader for extracting clean, readable page content from URLs.

What it does:
- runs a web search,
- retrieves the top results,
- fetches readable page text,
- truncates documents to keep outputs manageable,
- skips duplicate URLs,
- and returns both the normalized search results and extracted documents.

This is useful when:
- the answer depends on recent developments,
- the question is jurisdiction-specific and not in the local corpus,
- or the agent needs direct access to public web sources.

---

### 3. EU regulation MCP tools
The agent connects to an EU regulations MCP server and exposes a filtered set of tools for tasks such as:
- searching regulations,
- retrieving definitions,
- checking applicability,
- comparing requirements,
- and retrieving specific articles.

This is especially useful for:
- EU AI Act questions,
- applicability analysis,
- and article-level research.

---

### 4. USA regulation MCP tools
The agent also connects to a USA regulations MCP server and exposes tools for:
- searching regulations,
- checking applicability,
- comparing requirements,
- mapping controls,
- retrieving evidence requirements,
- generating compliance action items,
- and retrieving sections.

This makes the agent useful for structured research on USA AI governance and compliance materials.

---

### 5. Dutch law MCP tools
For Dutch legal questions, the agent includes a filtered Dutch-law MCP toolset that supports tasks such as:
- searching legislation,
- retrieving provisions,
- checking whether a provision is still current,
- retrieving Dutch implementations,
- validating EU compliance relationships,
- and tracing EU legal basis.

This is particularly helpful for Dutch statutory validation and cross-checking EU-derived obligations.

---

### 6. Automotive compliance MCP tools
The agent includes an automotive-focused MCP toolset for:
- listing sources,
- searching requirements,
- and listing work products.

This is useful for questions that sit at the intersection of:
- AI regulation,
- vehicle compliance,
- cybersecurity requirements,
- software updates,
- and evidence artifacts in automotive workflows.




