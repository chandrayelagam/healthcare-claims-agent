from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
from typing import List

_model = None
_index = None
_chunks: List[str] = []


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def index_policies(policy_dir: str = "policies/") -> None:
    """Chunk and embed all .md policy documents into a FAISS index."""
    global _index, _chunks
    model = _load_model()
    raw_chunks: List[str] = []

    for path in Path(policy_dir).glob("*.md"):
        text = path.read_text()
        # 512-char chunks with 64-char overlap
        for i in range(0, len(text), 448):
            chunk = text[i : i + 512].strip()
            if len(chunk) > 100:
                raw_chunks.append(chunk)

    if not raw_chunks:
        return

    embeddings = model.encode(raw_chunks, show_progress_bar=False)
    dim = embeddings.shape[1]
    _index = faiss.IndexFlatL2(dim)
    _index.add(np.array(embeddings, dtype="float32"))
    _chunks = raw_chunks
    print(f"[RAG] Indexed {len(_chunks)} chunks from {policy_dir}")


def retrieve_policies(query: str, top_k: int = 3) -> List[str]:
    """Retrieve the top-k policy chunks most relevant to the query."""
    global _index, _chunks
    if _index is None:
        index_policies()
    if not _chunks:
        return ["No policy documents indexed."]

    model = _load_model()
    query_vec = model.encode([query])
    _, indices = _index.search(np.array(query_vec, dtype="float32"), top_k)
    return [_chunks[i] for i in indices[0] if i < len(_chunks)]
