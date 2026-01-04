# import faiss
# import pickle
# import numpy as np
# import re
# import os
# from sentence_transformers import SentenceTransformer
# import ollama

# # -------------------------------
# # Intent keywords
# # -------------------------------
# SUMMARY_KEYWORDS = [
#     "what is inside",
#     "overview",
#     "summary",
#     "about this pdf",
#     "what does this pdf contain"
# ]

# PAGE_KEYWORDS = [
#     "explain page",
#     "describe page",
#     "what is on page"
# ]

# # -------------------------------
# # Load embedding model once
# # -------------------------------
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # -------------------------------
# # Utility: Load metadata safely
# # -------------------------------
# def load_meta(pdf):
#     meta_path = f"data/faiss_indexes/{pdf}.meta"
#     if not os.path.exists(meta_path):
#         return None
#     with open(meta_path, "rb") as f:
#         return pickle.load(f)

# # -------------------------------
# # PDF Summary Function
# # -------------------------------
# def summarize_pdfs(selected_pdfs):
#     all_text = []

#     for pdf in selected_pdfs:
#         meta = load_meta(pdf)
#         if not meta:
#             continue
#         for c in meta:
#             all_text.append(c.get("text", ""))

#     if not all_text:
#         return "❌ No content available to summarize."

#     combined_text = "\n".join(all_text[:3000])

#     prompt = f"""
# Summarize the following PDF content clearly and briefly.
# Do NOT add information outside the document.

# Content:
# {combined_text}
# """

#     response = ollama.chat(
#         model="phi",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     return response["message"]["content"]

# # -------------------------------
# # Page-wise Explanation
# # -------------------------------
# def explain_page(page_number, selected_pdfs):
#     page_text = []

#     for pdf in selected_pdfs:
#         meta = load_meta(pdf)
#         if not meta:
#             continue

#         for c in meta:
#             if c.get("page") == page_number:
#                 page_text.append(c.get("text", ""))

#     if not page_text:
#         return f"❌ No content found on page {page_number}."

#     combined_text = "\n".join(page_text[:3000])

#     prompt = f"""
# Explain the following page content clearly and in simple terms:

# {combined_text}
# """

#     response = ollama.chat(
#         model="phi",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     return response["message"]["content"]

# # -------------------------------
# # Main Query Function
# # -------------------------------
# def query_pdfs(question, selected_pdfs, k=4):
#     q_lower = question.lower()

#     # 🔹 PAGE MODE
#     if any(key in q_lower for key in PAGE_KEYWORDS):
#         page_nums = re.findall(r"\d+", q_lower)
#         if page_nums:
#             return explain_page(int(page_nums[0]), selected_pdfs)

#     # 🔹 SUMMARY MODE
#     if any(key in q_lower for key in SUMMARY_KEYWORDS):
#         return summarize_pdfs(selected_pdfs)

#     # 🔹 NORMAL QA MODE (FAISS)
#     all_chunks = []
#     q_emb = model.encode(
#         [question],
#         normalize_embeddings=True
#     ).astype("float32")

#     missing = []

#     for pdf in selected_pdfs:
#         index_path = f"data/faiss_indexes/{pdf}.index"
#         meta_path = f"data/faiss_indexes/{pdf}.meta"

#         if not os.path.exists(index_path) or not os.path.exists(meta_path):
#             missing.append(pdf)
#             continue

#         index = faiss.read_index(index_path)
#         meta = load_meta(pdf)

#         if not meta:
#             continue

#         D, I = index.search(q_emb, k)

#         for idx in I[0]:
#             if idx < len(meta):
#                 all_chunks.append(meta[idx])

#     if missing:
#         return (
#             "❌ Some selected PDFs are not indexed.\n"
#             f"Missing indexes: {', '.join(missing)}\n"
#             "Please re-upload them in the app."
#         )

#     if not all_chunks:
#         return "❌ Answer not found in selected PDFs."

#     # -------------------------------
#     # Build context with metadata
#     # -------------------------------
#     context = "\n".join(
#         f"(PDF: {c.get('pdf_name','Unknown')} | "
#         f"Section: {c.get('section','Unknown')} | "
#         f"Page {c.get('page','?')}): {c.get('text','')}"
#         for c in all_chunks
#     )

#     prompt = f"""
# Answer ONLY using the information below.
# Do NOT add external knowledge.

# Context:
# {context}

# Question:
# {question}
# """

#     response = ollama.chat(
#         model="phi",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     answer = response["message"]["content"]

#     # -------------------------------
#     # Confidence Score
#     # -------------------------------
#     confidence = min(95, 50 + len(set(c["page"] for c in all_chunks)) * 8)

#     # -------------------------------
#     # Build citation section
#     # -------------------------------
#     sources = sorted({
#         f"{c.get('pdf_name','Unknown')} | "
#         f"{c.get('section','Unknown')} | "
#         f"Page {c.get('page','?')}"
#         for c in all_chunks
#     })

#     citation_text = "\n".join(f"• {s}" for s in sources)

#     return f"""{answer}

# 📊 Confidence: {confidence}%

# 📌 Sources:
# {citation_text}
# """
# using groq instead of ollama

import faiss
import pickle
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------
# Intent keywords
# -------------------------------
SUMMARY_KEYWORDS = [
    "what is inside",
    "overview",
    "summary",
    "about this pdf",
    "what does this pdf contain"
]

PAGE_KEYWORDS = [
    "explain page",
    "describe page",
    "what is on page"
]

# -------------------------------
# Load embedding model
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Utility: Load metadata safely
# -------------------------------
def load_meta(pdf):
    path = f"data/faiss_indexes/{pdf}.meta"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

# -------------------------------
# LLM Call (Groq-safe)
# -------------------------------
def call_llm(prompt):
    if not prompt or len(prompt.strip()) < 30:
        return "❌ Not enough content to answer."

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # ✅ UPDATED MODEL
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Answer strictly using the given context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=512
    )

    return response.choices[0].message.content.strip()

# -------------------------------
# PDF Summary
# -------------------------------
def summarize_pdfs(selected_pdfs):
    all_text = []

    for pdf in selected_pdfs:
        meta = load_meta(pdf)
        if not meta:
            continue
        for c in meta:
            all_text.append(c.get("text", ""))

    if not all_text:
        return "❌ No content available to summarize."

    combined_text = "\n".join(all_text)[:4000]

    prompt = f"""
Summarize the following document clearly and briefly.
Do NOT add information outside the document.

Content:
{combined_text}
"""

    return call_llm(prompt)

# -------------------------------
# Page Explanation
# -------------------------------
def explain_page(page_number, selected_pdfs):
    page_text = []

    for pdf in selected_pdfs:
        meta = load_meta(pdf)
        if not meta:
            continue

        for c in meta:
            if c.get("page") == page_number:
                page_text.append(c.get("text", ""))

    if not page_text:
        return f"❌ No content found on page {page_number}."

    combined_text = "\n".join(page_text)[:4000]

    prompt = f"""
Explain the following page content in simple terms:

{combined_text}
"""

    return call_llm(prompt)

# -------------------------------
# Main Query Function
# -------------------------------
def query_pdfs(question, selected_pdfs, k=4):
    q_lower = question.lower()

    # 🔹 Page intent
    if any(key in q_lower for key in PAGE_KEYWORDS):
        nums = re.findall(r"\d+", q_lower)
        if nums:
            return explain_page(int(nums[0]), selected_pdfs)

    # 🔹 Summary intent
    if any(key in q_lower for key in SUMMARY_KEYWORDS):
        return summarize_pdfs(selected_pdfs)

    # 🔹 FAISS QA
    all_chunks = []
    q_emb = model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    missing = []

    for pdf in selected_pdfs:
        index_path = f"data/faiss_indexes/{pdf}.index"
        meta_path = f"data/faiss_indexes/{pdf}.meta"

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            missing.append(pdf)
            continue

        index = faiss.read_index(index_path)
        meta = load_meta(pdf)

        D, I = index.search(q_emb, k)
        for idx in I[0]:
            if idx < len(meta):
                all_chunks.append(meta[idx])

    if missing:
        return f"❌ Missing indexes: {', '.join(missing)}"

    if not all_chunks:
        return "❌ Answer not found in selected PDFs."

    context = "\n".join(
        f"(PDF: {c.get('pdf_name','Unknown')} | "
        f"Section: {c.get('section','Unknown')} | "
        f"Page {c.get('page','?')}): {c.get('text','')}"
        for c in all_chunks
    )[:4000]

    prompt = f"""
Answer ONLY using the information below.
Do NOT add external knowledge.

Context:
{context}

Question:
{question}
"""

    answer = call_llm(prompt)

    confidence = min(95, 50 + len(set(c["page"] for c in all_chunks)) * 8)

    sources = sorted({
        f"{c.get('pdf_name','Unknown')} | {c.get('section','Unknown')} | Page {c.get('page','?')}"
        for c in all_chunks
    })

    return f"""{answer}

📊 Confidence: {confidence}%

📌 Sources:
""" + "\n".join(f"• {s}" for s in sources)
