# RAG infrastructure
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models


# ----------------------------
# Config
# ----------------------------

DATASETS = [
    "github_datatalksclub",
    "granite_docling_pdfs",
    "youtube_podcasts",
]


# ----------------------------
# Chunking
# ----------------------------

def chunk_text(text: str, chunk_size: int = 5000, overlap: int = 500) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, start + 1)

    return chunks


# ----------------------------
# Source reading
# ----------------------------

@dataclass
class SourceChunk:
    chunk_id: str              # stable logical chunk id
    point_id: str              # Qdrant point id (UUID string)
    text: str
    source_path: str
    source_type: str
    title: str
    chunk_index: int
    metadata: dict[str, Any]   # loaded from matching JSON


def iter_txt_files(data_root: Path) -> Iterable[Path]:
    for dataset in DATASETS:
        txt_dir = data_root / dataset / "txt"
        if txt_dir.exists():
            yield from txt_dir.rglob("*.txt")


def get_dataset_name(txt_path: Path, data_root: Path) -> str:
    for dataset in DATASETS:
        txt_dir = data_root / dataset / "txt"
        try:
            txt_path.relative_to(txt_dir)
            return dataset
        except ValueError:
            continue
    return "unknown"


def metadata_path_for_txt(txt_path: Path, data_root: Path) -> Path | None:
    """
    Map:
      data/<dataset>/txt/.../file.txt
    to:
      data/<dataset>/json/.../file.json
    """
    for dataset in DATASETS:
        txt_dir = data_root / dataset / "txt"
        json_dir = data_root / dataset / "json"

        try:
            rel = txt_path.relative_to(txt_dir)
            return json_dir / rel.with_suffix(".json")
        except ValueError:
            continue

    return None


def load_metadata_for_txt(txt_path: Path, data_root: Path) -> dict[str, Any]:
    json_path = metadata_path_for_txt(txt_path, data_root)
    if json_path is None or not json_path.exists():
        return {}

    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    # Keep only metadata. Do not store duplicate large text bodies from JSON.
    # Exclude certain metadata keys. 
    for key in ("text", "transcript_paragraphs", "transcript"): 
        raw.pop(key, None)

    return raw


def detect_source_type(txt_path: Path, data_root: Path) -> str:
    dataset = get_dataset_name(txt_path, data_root)

    if dataset == "youtube_podcasts":
        return "youtube_transcript"
    if dataset == "granite_docling_pdfs":
        return "docling_pdf"
    if dataset == "github_datatalksclub":
        return "github_text"

    return "local_text"


def infer_title(path: Path, metadata: dict[str, Any]) -> str:
    return (
        metadata.get("title")
        or metadata.get("video_title")
        or path.stem.replace("_", " ").replace("-", " ").strip()
    )


def build_chunks(
    data_root: Path,
    chunk_size: int = 5000,
    overlap: int = 500,
) -> list[SourceChunk]:
    out: list[SourceChunk] = []

    for txt_path in iter_txt_files(data_root):
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        metadata = load_metadata_for_txt(txt_path, data_root)
        source_type = detect_source_type(txt_path, data_root)
        title = infer_title(txt_path, metadata)

        for i, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            raw_id = f"{txt_path.resolve()}::{i}"

            # UUID for Qdrant point id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))

            # Optional stable logical id
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()

            out.append(
                SourceChunk(
                    chunk_id=chunk_id,
                    point_id=point_id,
                    text=chunk,
                    source_path=str(txt_path.resolve()),
                    source_type=source_type,
                    title=title,
                    chunk_index=i,
                    metadata=metadata,
                )
            )

    return out


# ----------------------------
# Jina embedder
# ----------------------------

class JinaEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "jina-embeddings-v5-text",
        base_url: str = "https://api.jina.ai/v1/embeddings",
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": texts,
                    "normalized": True,
                    "embedding_type": "float",
                },
            )
            resp.raise_for_status()
            data = resp.json()["data"]

        data = sorted(data, key=lambda x: x["index"])
        return [item["embedding"] for item in data]


# ----------------------------
# Vector store
# ----------------------------

class LocalRAG:
    def __init__(self, qdrant_path: str, collection_name: str, embedder) -> None:
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        self.embedder = embedder

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def ensure_index(
        self,
        data_root: str,
        chunk_size: int = 5000,
        overlap: int = 500,
        batch_size: int = 64,
    ) -> bool:
        if self.collection_exists():
            return False

        self.rebuild_index(
            data_root=data_root,
            chunk_size=chunk_size,
            overlap=overlap,
            batch_size=batch_size,
        )
        return True

    def rebuild_index(
        self,
        data_root: str,
        chunk_size: int = 5000,
        overlap: int = 500,
        batch_size: int = 64,
    ) -> None:
        chunks = build_chunks(Path(data_root), chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise ValueError("No .txt files found to index.")

        # Embedding text only, not metadata
        probe_vec = self.embedder.embed([chunks[0].text])[0]
        vector_size = len(probe_vec)

        if self.collection_exists():
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            on_disk_payload=True,
        )

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]

            # IMPORTANT: only embed chunk text
            embeddings = self.embedder.embed([c.text for c in batch])

            points = []
            for c, vec in zip(batch, embeddings):
                points.append(
                    models.PointStruct(
                        id=c.point_id,
                        vector=vec,
                        payload={
                            "chunk_id": c.chunk_id,
                            "text": c.text,
                            "source_path": c.source_path,
                            "source_type": c.source_type,
                            "title": c.title,
                            "chunk_index": c.chunk_index,
                            "metadata": c.metadata,   # JSON metadata stored here
                        },
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search(self, query: str, limit: int = 6) -> list[dict]:
        qvec = self.embedder.embed([query])[0]

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=qvec,
            limit=limit,
            with_payload=True,
        )

        hits = []
        for p in result.points:
            payload = p.payload or {}
            hits.append(
                {
                    "score": float(p.score),
                    "text": payload.get("text", ""),
                    "source_path": payload.get("source_path", ""),
                    "source_type": payload.get("source_type", ""),
                    "title": payload.get("title", ""),
                    "chunk_index": payload.get("chunk_index", -1),
                    "metadata": payload.get("metadata", {}),
                }
            )
        return hits