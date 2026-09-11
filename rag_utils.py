"""
rag_utils.py
============
Local Retrieval-Augmented Generation (RAG) layer for the Course Content
Simplification Agent.

Responsibilities:
  - Load all .txt files from a knowledge_base/ folder
  - Chunk each file into overlapping ~400-word windows
  - Embed chunks locally with sentence-transformers (CPU-only, no API cost)
  - Given a query string, return the top-k most relevant chunks filtered by
    a minimum cosine-similarity threshold

No external API is called; everything runs on CPU.
"""

import os
import glob
from typing import List, Tuple, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Module-level state (lazy-initialised on first retrieve() call)
# ---------------------------------------------------------------------------
_model: SentenceTransformer = None          # sentence-transformer model
_chunk_records: List[Dict] = []             # list of {chunk, source, embedding}
_index_built: bool = False                  # flag to avoid re-building

# Default knowledge base folder (relative to project root)
DEFAULT_KB_FOLDER = "knowledge_base"

# Embedding model — runs on CPU, ~90 MB download on first use
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------

def load_knowledge_base(folder: str) -> List[Tuple[str, str]]:
    """
    Read every .txt file in *folder* and return a list of (text, filename) tuples.

    Parameters
    ----------
    folder : str
        Path to the directory containing .txt knowledge-base files.

    Returns
    -------
    List[Tuple[str, str]]
        Each tuple is (file_contents, filename_without_path).
    """
    pattern = os.path.join(folder, "*.txt")
    files = sorted(glob.glob(pattern))  # sorted for deterministic ordering

    if not files:
        raise FileNotFoundError(
            f"No .txt files found in '{folder}'. "
            "Add at least one glossary file to enable RAG grounding."
        )

    records = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
        filename = os.path.basename(filepath)
        records.append((text, filename))

    return records


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """
    Split *text* into overlapping word-count windows.

    Parameters
    ----------
    text : str
        The full text to chunk.
    chunk_size : int
        Target number of words per chunk (default 400).
    overlap : int
        Number of words shared between consecutive chunks (default 80).

    Returns
    -------
    List[str]
        List of chunk strings; each chunk is at most *chunk_size* words.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)  # advance by (chunk_size - overlap) words

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def _load_model() -> SentenceTransformer:
    """Load (or return cached) the sentence-transformer model on CPU."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return _model


def build_index(folder: str = DEFAULT_KB_FOLDER) -> None:
    """
    Load all .txt files from *folder*, chunk them, embed every chunk, and
    store the results in module-level state.

    This function is idempotent — calling it multiple times with the same
    folder is safe; it rebuilds only once unless reset manually.

    Parameters
    ----------
    folder : str
        Path to the knowledge-base directory (default: "knowledge_base").
    """
    global _chunk_records, _index_built

    if _index_built:
        return  # already built; skip re-computation

    model = _load_model()
    kb_files = load_knowledge_base(folder)

    records = []
    for text, filename in kb_files:
        chunks = chunk_text(text)
        for chunk in chunks:
            embedding = model.encode(chunk, normalize_embeddings=True)
            records.append({
                "chunk": chunk,
                "source": filename,
                "embedding": embedding,  # numpy array, already L2-normalised
            })

    _chunk_records = records
    _index_built = True


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = 4,
    min_score: float = 0.15,
    folder: str = DEFAULT_KB_FOLDER,
) -> List[Dict]:
    """
    Retrieve the most relevant knowledge-base chunks for *query*.

    Builds the index on the first call (lazy initialisation).

    Parameters
    ----------
    query : str
        The academic text the user wants to simplify; used as the retrieval query.
    top_k : int
        Maximum number of chunks to return (default 4).
    min_score : float
        Minimum cosine-similarity threshold; chunks below this are discarded
        (default 0.15 — filters out clearly unrelated content).
    folder : str
        Knowledge-base folder path, passed to build_index if not yet built.

    Returns
    -------
    List[Dict]
        List of result dicts, sorted descending by score, each containing:
          - "chunk"  : str   — the chunk text
          - "source" : str   — filename the chunk came from
          - "score"  : float — cosine similarity in [0, 1]
        May be empty if no chunk exceeds *min_score*.
    """
    # Ensure index is ready
    build_index(folder)

    if not _chunk_records:
        return []

    model = _load_model()

    # Encode query (normalised so cosine-sim == dot product)
    query_embedding = model.encode(query, normalize_embeddings=True)

    # Compute cosine similarities against all chunk embeddings
    results = []
    for record in _chunk_records:
        # Both vectors are L2-normalised, so dot product == cosine similarity
        score = float(np.dot(query_embedding, record["embedding"]))
        if score >= min_score:
            results.append({
                "chunk": record["chunk"],
                "source": record["source"],
                "score": score,
            })

    # Sort by score descending and return top-k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def reset_index() -> None:
    """
    Clear the in-memory index so it will be rebuilt on the next retrieve() call.
    Useful for testing or after adding new knowledge-base files at runtime.
    """
    global _chunk_records, _index_built
    _chunk_records = []
    _index_built = False
