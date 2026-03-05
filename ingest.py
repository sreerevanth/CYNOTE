import os
import warnings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

warnings.filterwarnings("ignore")

DATA_PATH = "data/raw_docs"
DB_PATH = "embeddings"


def clean_text(text: str) -> str:
    """
    Remove repeated lines, heading spam, and OCR noise
    """
    lines = text.split("\n")
    cleaned_lines = []
    seen = set()

    for line in lines:
        line = line.strip()

        # skip very short junk
        if len(line) < 5:
            continue

        # remove duplicates (case-insensitive)
        lower = line.lower()
        if lower in seen:
            continue

        seen.add(lower)
        cleaned_lines.append(line)

    return " ".join(cleaned_lines)


def ingest_documents():
    print("Looking inside:", os.path.abspath(DATA_PATH))

    if not os.path.exists(DATA_PATH):
        print("❌ data/raw_docs folder not found.")
        return

    files = os.listdir(DATA_PATH)
    print("Files found:", files)

    documents = []

    for file in files:
        if file.lower().endswith(".pdf"):
            print(f"📄 Loading: {file}")
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            pages = loader.load()

            for page in pages:
                page.page_content = clean_text(page.page_content)
                documents.append(page)

    if not documents:
        print("❌ No documents loaded. Exiting.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)
    print(f"🔹 Total chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(DB_PATH, exist_ok=True)
    db.save_local(DB_PATH)

    print("✅ Ingestion complete. Knowledge stored.")


if __name__ == "__main__":
    ingest_documents()
