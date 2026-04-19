# Define web-search tool using the [Brave Search API](https://brave.com/search/api/) and the [Jina Reader API](https://jina.ai/reader/)

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional
from pydantic import BaseModel, Field, HttpUrl
from pydantic_ai import RunContext

from .dependencies import Deps

class BraveResult(BaseModel):
    """A single web search result returned by Brave Search."""
    title: str
    url: HttpUrl
    description: Optional[str] = None

class WebDocument(BaseModel):
    """Readable webpage content extracted from a search result URL."""
    url: HttpUrl
    title: str = ""
    description: Optional[str] = None
    text: str
    fetched_at_utc: str
    sha256: str
    meta: dict[str, Any] = Field(default_factory=dict)

class WebSearchAndReadOutput(BaseModel):
    """Structured output containing search results, extracted documents, and per-URL errors."""
    query: str
    search_results: list[BraveResult]
    documents: list[WebDocument]
    errors: list[dict[str, str]]

def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def sha256_text(s: str) -> str:
    """Return the SHA-256 hex digest of a text string."""
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def _truncate(s: str, max_chars: int) -> str:
    """Truncate a string to at most `max_chars` characters."""
    if max_chars <= 0:
        return s
    return s[:max_chars]

# Agent web-search tool
def web_search_and_read(
    ctx: RunContext[Deps],
    query: str,
    num_results: int = 5,
    max_chars_per_doc: int = 20000,
    use_readerlm_v2: bool = True,
) -> WebSearchAndReadOutput:
    """
    Search the web for a query and extract readable text from the top results.
    This tool first uses Brave Search to retrieve a small set of relevant web
    pages for the given query. It then visits each result URL through Jina Reader
    to obtain cleaned, readable page content that is easier for the agent to use
    than raw HTML.

    The tool returns:
    - the original query,
    - the normalized Brave search results,
    - the extracted text documents for URLs that were read successfully,
    - and a list of per-URL errors for any pages that could not be processed.

    To keep the output manageable, each extracted document is truncated to
    `max_chars_per_doc` characters. Duplicate URLs are skipped, and individual
    page-reading failures do not stop the overall tool run.

    Args:
        ctx: Run context containing shared dependencies such as the Brave Search
            client, the Jina Reader client, and small runtime settings.
        query: The web search query to run.
        num_results: The maximum number of Brave search results to retrieve.
        max_chars_per_doc: The maximum number of characters to keep from each
            extracted document.
        use_readerlm_v2: Whether to ask Jina Reader to use ReaderLM-v2 for
            potentially better extraction quality.

    Returns:
        A `WebSearchAndReadOutput` object containing the search results,
        extracted documents, and any errors encountered while processing URLs.
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    if num_results < 1:
        raise ValueError("num_results must be >= 1")

    if max_chars_per_doc < 1:
        raise ValueError("max_chars_per_doc must be >= 1")

    deps = ctx.deps

    brave_raw = deps.brave.search(query=query, count=num_results)
    search_results = [BraveResult(**item) for item in brave_raw]

    documents: list[WebDocument] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()

    for result in search_results:
        url = str(result.url)

        if url in seen:
            continue
        seen.add(url)

        time.sleep(deps.per_request_sleep_s)

        try:
            resp = deps.jina.read_url(
                url,
                use_readerlm_v2=use_readerlm_v2,
                bypass_cache=True,
            )

            if not isinstance(resp, dict):
                raise ValueError("Unexpected Jina response type")

            data = resp.get("data", {})
            if not isinstance(data, dict):
                raise ValueError("Unexpected Jina response payload")

            title = (data.get("title") or result.title or url).strip()
            content = (data.get("content") or "").strip()

            if not content:
                raise ValueError("Empty content returned by Jina Reader")

            content = _truncate(content, max_chars_per_doc)
            content_hash = sha256_text(content)

            documents.append(
                WebDocument(
                    url=result.url,
                    title=title,
                    description=result.description,
                    text=content,
                    fetched_at_utc=utc_now_iso(),
                    sha256=content_hash,
                    meta={
                        "jina_status": resp.get("status"),
                        "jina_code": resp.get("code"),
                    },
                )
            )

        except Exception as e:
            errors.append(
                {
                    "url": url,
                    "error": str(e),
                }
            )

    return WebSearchAndReadOutput(
        query=query,
        search_results=search_results,
        documents=documents,
        errors=errors,
    )