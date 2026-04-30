import os
import warnings
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw_docs")
DB_PATH = os.path.join(BASE_DIR, "embeddings")


def clean_text(text: str) -> str:
    """
    Remove repeated lines, heading spam, and OCR noise.
    Uses dict.fromkeys for fast, order-preserving deduplication.
    """
    lines = list(dict.fromkeys(
        line.strip() for line in text.split("\n") if len(line.strip()) > 5
    ))
    return " ".join(lines)


def load_pdf(file: str) -> list:
    """Load and clean a single PDF. Designed for parallel execution."""
    loader = PyPDFLoader(os.path.join(DATA_PATH, file))
    pages = loader.load()
    for page in pages:
        page.page_content = clean_text(page.page_content)
    return pages


def ingest_documents():
    print("Looking inside:", os.path.abspath(DATA_PATH))

    # ⚡ Skip re-ingestion if embeddings already exist
    if os.path.exists(os.path.join(DB_PATH, "index.faiss")):
        print("⚡ Using existing embeddings. Skipping ingestion.")
        return

    if not os.path.exists(DATA_PATH):
        print("❌ data/raw_docs folder not found.")
        return

    files = os.listdir(DATA_PATH)
    print("Files found:", files)

    pdf_files = [f for f in files if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDF files found. Exiting.")
        return

    # 🔥 Parallel PDF loading
    print(f"📄 Loading {len(pdf_files)} PDF(s) in parallel...")
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_pdf, pdf_files))

    documents = [page for sublist in results for page in sublist]

    if not documents:
        print("❌ No documents loaded. Exiting.")
        return

    # 🧠 Larger chunks = fewer embeddings = faster
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    print(f"🔹 Total chunks created: {len(chunks)}")

    # ✅ Sanity check
    print("\n--- Sanity Check: First Chunk ---")
    print(chunks[0].page_content[:500])
    print(f"Metadata: {chunks[0].metadata}")
    print("---------------------------------\n")

    # 🚀 GPU if available, else CPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",       # 🧬 Faster + better than MiniLM
        model_kwargs={"device": device}
    )

    # ⚡ Batch embeddings via from_texts (avoids Document overhead)
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    db = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    os.makedirs(DB_PATH, exist_ok=True)
    db.save_local(DB_PATH)

    print("✅ Ingestion complete. Knowledge stored.")


if __name__ == "__main__":
    ingest_documents()
