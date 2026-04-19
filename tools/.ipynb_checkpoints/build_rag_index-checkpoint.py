import os
from .rag_local import JinaEmbedder, LocalRAG

# Retrieval augmented generation set-up
# One-time RAG index construction
jina_api_key = os.environ["JINA_API_KEY"]
embedder = JinaEmbedder(api_key=jina_api_key, 
                        model=os.getenv("JINA_EMBED_MODEL", "jina-embeddings-v5-text"))
rag = LocalRAG(qdrant_path=".qdrant_rag", collection_name="local_dataset", embedder=embedder)
rag.rebuild_index(data_root="data", chunk_size=1200, overlap=200, batch_size=64)
print("RAG index done.")