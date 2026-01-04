import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle

# ---------------------------
# Load embedding model once
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Build FAISS index
# ---------------------------
def build_faiss(pdf_name, chunks, force_rebuild=False):
    os.makedirs("data/faiss_indexes", exist_ok=True)

    index_path = f"data/faiss_indexes/{pdf_name}.index"
    meta_path = f"data/faiss_indexes/{pdf_name}.meta"

    # ---------------------------
    # Skip rebuild if already exists
    # ---------------------------
    if not force_rebuild and os.path.exists(index_path) and os.path.exists(meta_path):
        return

    # ---------------------------
    # Safety check
    # ---------------------------
    if not chunks:
        return

    texts = [c["text"] for c in chunks if c.get("text")]

    if not texts:
        return

    # ---------------------------
    # Generate embeddings
    # ---------------------------
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    dim = embeddings.shape[1]

    # ---------------------------
    # Build FAISS index
    # ---------------------------
    index = faiss.IndexFlatL2(dim)
    index.add(np.asarray(embeddings, dtype="float32"))

    # ---------------------------
    # Save index
    # ---------------------------
    faiss.write_index(index, index_path)

    # ---------------------------
    # Ensure metadata consistency
    # ---------------------------
    for c in chunks:
        c["pdf_name"] = pdf_name
        if "section" not in c:
            c["section"] = "Unknown"

    # ---------------------------
    # Save metadata
    # ---------------------------
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)
