import streamlit as st
from pdf_processor import extract_pdf_text
from vector_store import build_faiss
from qa_engine import query_pdfs
import os
import pickle
import re

# ---------------------------
# Helpers
# ---------------------------
def safe_pdf_name(name):
    name = name.replace(".pdf", "")
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return name

def sync_session_with_disk():
    os.makedirs("data/faiss_indexes", exist_ok=True)
    st.session_state.indexed_pdfs = sorted([
        f.replace(".index", "")
        for f in os.listdir("data/faiss_indexes")
        if f.endswith(".index")
    ])

def clear_indexes():
    if os.path.exists("data/faiss_indexes"):
        for f in os.listdir("data/faiss_indexes"):
            os.remove(os.path.join("data/faiss_indexes", f))
    st.session_state.indexed_pdfs = []

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Advanced Multi-PDF AI Q&A",
    layout="wide"
)

st.title("📘 Advanced Multi-PDF AI Q&A System")

# ---------------------------
# Session State Init
# ---------------------------
if "indexed_pdfs" not in st.session_state:
    st.session_state.indexed_pdfs = []

sync_session_with_disk()

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("🛠 Controls")

if st.sidebar.button("🗑 Clear all PDF indexes"):
    clear_indexes()
    st.sidebar.success("Indexes cleared. Re-upload PDFs.")

st.sidebar.divider()
st.sidebar.header("📂 Select PDFs")

# ---------------------------
# Upload PDFs
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Indexing PDFs..."):
        for file in uploaded_files:
            pdf_name = safe_pdf_name(file.name)
            pages = extract_pdf_text(file)

            # Always rebuild to reflect latest parsing logic
            build_faiss(pdf_name, pages, force_rebuild=True)

    sync_session_with_disk()
    st.success("PDFs indexed successfully!")

# ---------------------------
# PDF Selection
# ---------------------------
selected_pdfs = st.sidebar.multiselect(
    "Choose PDFs to query",
    st.session_state.indexed_pdfs,
    key="pdf_selector"
)

# ---------------------------
# Question Input
# ---------------------------
question = st.text_input(
    "Ask a question (e.g. *What is inside the PDF?*, *Explain page 10*, *Explain continuous signal*)"
)

# ---------------------------
# Ask Button
# ---------------------------
if st.button("Ask AI"):
    if not selected_pdfs:
        st.warning("Please select at least one PDF.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = query_pdfs(question, selected_pdfs)

        # ---------------------------
        # Main Answer
        # ---------------------------
        st.success(answer)

        # ---------------------------
        # Highlight Source Text
        # ---------------------------
        with st.expander("🔍 View source text used by AI"):
            for pdf in selected_pdfs:
                meta_path = f"data/faiss_indexes/{pdf}.meta"
                if not os.path.exists(meta_path):
                    continue

                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)

                for c in meta[:8]:
                    st.markdown(
                        f"""
**📄 {c.get('pdf_name','')}  
📌 Section: {c.get('section','Unknown')}  
📖 Page: {c['page']}**

{c['text']}
---
"""
                    )
