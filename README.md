# ⚡ CYNOTE — AI Study Engine

> **Ask anything from your notes. Get exam-ready answers via Gemma 3 27B.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=flat-square)
![Gemma](https://img.shields.io/badge/Gemma_3-27B-orange?style=flat-square&logo=google)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## 🧠 What is CYNOTE?

CYNOTE is an **offline-first, exam-oriented RAG (Retrieval Augmented Generation)** study assistant. Drop in your PDF notes, ask any question, and get structured answers — 2 marks, 5 marks, 10 marks, or full exam notes — powered by **Google's Gemma 3 27B** model via Google AI Studio.

No hallucination. No outside knowledge. Just your notes, intelligently retrieved and answered.

---

## ✨ Features

- 🔍 **Semantic Search** — FAISS vector store finds the most relevant chunks from your PDFs
- 🧠 **Gemma 3 27B** — Google's open model via AI Studio API (free tier)
- 🎯 **Exam Modes** — 2 Marks, 5 Marks, 10 Marks, Show All Related, Build Exam Notes
- 🔒 **Privacy First** — your PDFs are embedded and stored locally, never uploaded anywhere
- ⚡ **Zero Hallucination** — model is instructed to answer strictly from your notes
- 🌗 **Premium UI** — dark cyber theme built with custom CSS in Streamlit

---

## 🗂️ Project Structure

```
cynote/
│
├── app.py              # Streamlit frontend + Gemma 3 API integration
├── ingest.py           # PDF ingestion → chunking → FAISS embedding
├── query.py            # CLI query mode (no UI)
├── requirements.txt    # Python dependencies
│
├── data/
│   └── raw_docs/       # 📁 Drop your PDF notes here
│
└── embeddings/         # Auto-generated FAISS vector index
    ├── index.faiss
    └── index.pkl
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install google-genai
```

### 3. Add your PDF notes

Drop your study PDFs into:
```
data/raw_docs/
```

### 4. Ingest and index your PDFs

```bash
python ingest.py
```

> Run this every time you add new PDFs. Builds the FAISS vector index.

### 5. Get your free Gemma 3 API key

1. Go to **[aistudio.google.com/apikey](https://aistudio.google.com/apikey)**
2. Sign in with Google → Create API Key
3. Copy the `AIza...` key

### 6. Launch the app

```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501` — paste your API key in the sidebar and start asking!

---

## 🎓 Answer Modes

| Mode | Description |
|---|---|
| **2 Marks — Brief** | Concise 3-5 sentence answer |
| **5 Marks — Detailed** | Definition + explanation + example |
| **10 Marks — In-depth** | Full answer with intro, sub-points, conclusion |
| **Show All Related** | All relevant concepts from notes |
| **Build Exam Notes** | Structured revision notes with headings & key terms |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit + Custom CSS |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (local) |
| **PDF Parsing** | LangChain + PyPDF |
| **LLM** | Gemma 3 27B via Google AI Studio |
| **RAG Framework** | LangChain Community |

---

## ⚙️ CLI Mode (No UI)

You can also query directly from terminal:

```bash
python query.py
```

Edit the `question` variable inside `query.py` to change the query.

---

## 📦 Requirements

```
langchain
langchain-community
langchain-text-splitters
faiss-cpu
sentence-transformers
pypdf
streamlit
google-genai
```

---

## 🔮 Roadmap

- [ ] Upload PDFs directly from the UI
- [ ] Multi-subject filtering
- [ ] Export answers as PDF / Word
- [ ] Chat history within session
- [ ] Re-index button in sidebar
- [ ] Similarity score display per source chunk

---

## 🤝 Built By

Made with 🔥 by the Innovation Club.  
Powered by LangChain · FAISS · Gemma 3 27B · Streamlit.
