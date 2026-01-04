# 📘 DocQuery-AI  
**Multi-PDF AI Question Answering System (Local + Cloud)**

DocQuery-AI is an advanced **AI-powered document intelligence system** that allows users to upload, select, and query multiple PDF documents using **semantic search and Retrieval-Augmented Generation (RAG)**.

It supports **both offline (local LLM)** and **online (cloud LLM)** modes, ensuring flexibility for development, privacy, and deployment.

Answers are **strictly grounded in the uploaded PDFs**, with **page-level and section-level citations** to prevent hallucinations.

---

## 🌐 Live Demo (Cloud Mode)

🚀 **DocQuery-AI is live here:**  
👉 https://docquery-ai-o9hwz6rjgeah75ytyv6l9y.streamlit.app/

> ⚡ This live deployment uses **Groq LLM (`llama-3.1-8b-instant`)** for fast, free cloud inference.

---

## 🧠 LLM Modes Supported

### 🔹 Cloud Mode (Current Live Demo)
- **LLM:** Groq – `llama-3.1-8b-instant`
- **Inference:** Cloud-based
- **Cost:** Free tier (no billing)
- **Best for:** Deployment, demos, sharing

### 🔹 Offline Mode (Local Setup)
- **LLM:** Ollama (e.g., Phi, Mistral)
- **Inference:** Fully local
- **Internet:** Not required
- **Best for:** Privacy-focused & offline use

---

## 🚀 Key Features

- 📂 **Multi-PDF Upload & Selection**
- 🔍 **Semantic Search using FAISS**
- 🧠 **LLM-powered Answers (Groq or Ollama)**
- 📄 **Page-wise & Section-wise Explanations**
- 🧾 **Source Citations (PDF | Section | Page)**
- 📊 **Confidence Scoring**
- 🛡️ **Hallucination Control (PDF-grounded answers only)**
- ⚡ **Fast & Lightweight UI (Streamlit)**
- 💸 **No mandatory billing required**

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


