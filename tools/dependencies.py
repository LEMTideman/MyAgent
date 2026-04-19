from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Dict
from .rag_local import JinaEmbedder, LocalRAG

import requests

# -----------------------------
# Brave Web Search client
# -----------------------------

class BraveSearchClient:
    """Client for querying Brave Web Search and returning normalized search results."""
    def __init__(
        self,
        api_key: str,
        session: Optional[requests.Session] = None,
        timeout_s: int = 30,
    ):
        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout_s = timeout_s
        self.endpoint = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, count: int = 10) -> List[Dict[str, str]]:
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        }

        params = {"q": query, "count": count}

        r = self.session.get(
            self.endpoint,
            params=params,
            headers=headers,
            timeout=self.timeout_s,
        )
        r.raise_for_status()

        data = r.json()
        results = data["web"]["results"]

        out: List[Dict[str, str]] = []
        for item in results:
            url = item.get("url")
            title = item.get("title") or ""
            desc = item.get("description") or ""

            if url:
                out.append({"url": url, "title": title, "description": desc})

        return out

# -----------------------------
# Jina Webpage Reader client
# -----------------------------

class JinaReaderClient:
    """Client for extracting readable webpage content from a URL using Jina Reader."""
    def __init__(self, api_key: str, timeout_s: int = 60):
        # Choose the Jina Reader base URL.
        self.base = "https://r.jina.ai"
        # self.base = "https://eu.r.jina.ai" if eu else "https://r.jina.ai"

        # API key used for Authorization: Bearer <key>
        self.api_key = api_key

        # How long we wait (in seconds) before giving up on the HTTP request.
        self.timeout_s = timeout_s

    def read_url(self, url: str, use_readerlm_v2: bool = False) -> dict:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        if use_readerlm_v2:
            headers["X-Respond-With"] = "readerlm-v2"

        r = requests.post(
            f"{self.base}/",
            headers=headers,
            json={"url": url},
            timeout=self.timeout_s,
        )
        r.raise_for_status()

        return r.json()

# -----------------------------
# Dependencies
# -----------------------------

@dataclass
class Deps:
    """Container for shared external clients and runtime settings used by tools."""
    brave: BraveSearchClient
    jina: JinaReaderClient
    rag: LocalRAG
    per_request_sleep_s: float = 0.25


def build_deps() -> Deps:
    """Build and return the shared dependencies object from environment variables."""
    brave_key = os.environ["BRAVE_SEARCH_API_KEY"]
    jina_key = os.environ["JINA_API_KEY"]

    embedder = JinaEmbedder(api_key=os.environ["JINA_API_KEY"],
                            model=os.getenv("JINA_EMBED_MODEL", "jina-embeddings-v5-text-small"))
    session = requests.Session()

    return Deps(
        brave=BraveSearchClient(api_key=brave_key, session=session, timeout_s=30),
        jina=JinaReaderClient(api_key=jina_key, timeout_s=90),
        rag=LocalRAG(qdrant_path=".qdrant_rag", collection_name="local_dataset", embedder=embedder),
        per_request_sleep_s=0.25,
    )