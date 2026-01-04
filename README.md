# 📘 DocQuery-AI  
**Offline Multi-PDF AI Question Answering System**

DocQuery-AI is an advanced **offline AI-powered document intelligence system** that allows users to upload, select, and query multiple PDF documents using **semantic search and a local LLM**, without any cloud APIs or billing.

Built with a **Retrieval-Augmented Generation (RAG)** architecture, DocQuery-AI ensures accurate, source-grounded answers with page-level and section-level citations.

---

## 🚀 Key Features

- 📂 **Multi-PDF Upload & Selection**
- 🔍 **Semantic Search using FAISS**
- 🧠 **Local LLM (Ollama) – Fully Offline**
- 📄 **Page-wise & Section-wise Explanations**
- 🧾 **Source Citations (PDF | Section | Page)**
- 📊 **Confidence Scoring for Answers**
- 🛡️ **Hallucination Control (Answers only from PDFs)**
- ⚡ **Fast & Lightweight UI with Streamlit**
- 💸 **Zero API cost – No billing required**

---

## 🧠 System Architecture

PDFs
└── Text Extraction (PyMuPDF)
└── Section Detection (Font-based)
└── Embeddings (SentenceTransformers)
└── Vector Store (FAISS)
└── Query Retrieval
└── Local LLM (Ollama)
└── Answer + Sources


---

## 🧰 Tech Stack

| Layer | Technology |
|-----|-----------|
| UI | Streamlit |
| PDF Parsing | PyMuPDF |
| Embeddings | SentenceTransformers |
| Vector Database | FAISS |
| LLM | Ollama (phi / mistral) |
| Language | Python |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash

git clone https://github.com/Subhajit-Das-1/DocQuery-AI.git
cd DocQuery-AI
2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Install Ollama & Pull Model
ollama pull phi

4️⃣ Run the Application
streamlit run app.py
```

## 🖥️ How to Use

1. Upload one or more **PDF files**
2. Select PDFs from the **sidebar**
3. Ask questions such as:
   - *What is inside this PDF?*
   - *Explain page 10*
   - *Explain continuous-time signals*
4. View:
   - ✅ AI-generated answer
   - 📊 Confidence score
   - 📌 Source pages & sections
   - 🔍 Highlighted reference text

---

## 🧩 Future Enhancements

- 🧠 OCR support for scanned PDFs
- 📊 PDF comparison mode
- 📝 Auto-generated notes & summaries
- ❓ MCQ / exam question generator
- 📤 Export answers to **PDF / DOCX**

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Subhajit Das**  


