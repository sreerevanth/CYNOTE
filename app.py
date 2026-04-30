from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
import os, warnings, tempfile, logging

warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "embeddings")

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned, seen = [], set()
    for line in lines:
        line = line.strip()
        if len(line) < 5:
            continue
        lower = line.lower()
        if lower in seen:
            continue
        seen.add(lower)
        cleaned.append(line)
    return " ".join(cleaned)


def build_prompt(question, context, mode):
    instructions = {
        "2 Marks — Brief":     "Write a concise 2-mark answer in 3-5 sentences. Be direct and factual.",
        "5 Marks — Detailed":  "Write a 5-mark answer with definition, explanation, and an example.",
        "10 Marks — In-depth": "Write a comprehensive 10-mark answer with intro, detailed sub-points, examples, and conclusion.",
        "Show All Related":    "List all relevant concepts from the notes with brief explanations for each.",
        "Build Exam Notes":    "Build structured exam revision notes with headings, key points, and important terms.",
    }
    return f"""You are Helion AI, an expert exam study assistant. Answer ONLY using the provided notes.

INSTRUCTION: {instructions.get(mode, "Answer clearly and in detail.")}

NOTES FROM STUDENT DOCUMENTS:
{context}

QUESTION: {question}

Rules:
- Use ONLY information from the notes above
- If notes lack info, clearly say so
- Use bullet points and numbered lists where helpful
- Write in clean academic language
- Do NOT add outside information"""


def ask_gemma(api_key, prompt):
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")


# ── Routes ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    POST /upload  (multipart/form-data, field: file)
    Ingests the uploaded PDF into the FAISS store (merges with existing).
    Returns: { "chunks": <int> }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    try:
        # Save to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages  = loader.load()
        for page in pages:
            page.page_content = clean_text(page.page_content)
            # Tag source as the original filename
            page.metadata["source"] = f.filename

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks   = splitter.split_documents(pages)

        embeddings = get_embeddings()

        os.makedirs(DB_PATH, exist_ok=True)

        # Merge into existing FAISS store if it exists
        if os.path.exists(os.path.join(DB_PATH, "index.faiss")):
            db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(chunks)
        else:
            db = FAISS.from_documents(chunks, embeddings)

        db.save_local(DB_PATH)
        os.unlink(tmp_path)

        return jsonify({"chunks": len(chunks), "file": f.filename})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data     = request.get_json(force=True)
    question = data.get("question", "").strip()
    mode     = data.get("mode", "2 Marks — Brief")
    api_key  = data.get("api_key", "").strip()

    if not api_key:
        return jsonify({"error": "API key is required."}), 400
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400
    if not os.path.exists(DB_PATH):
        return jsonify({"error": "No documents ingested yet. Upload a PDF first."}), 500

    try:
        embeddings = get_embeddings()
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

        # ── Deep modes ──
        if mode in ["10 Marks — In-depth", "Build Exam Notes"]:
            topics_out = []
            topic_list = [
                "software specification",
                "software development",
                "software validation",
                "software evolution",
            ]
            for topic in topic_list:
                topic_docs = db.similarity_search(f"{question} {topic}", k=4)
                if not topic_docs:
                    continue
                topic_ctx   = "\n\n".join([d.page_content for d in topic_docs])
                answer_text = ask_gemma(api_key, build_prompt(f"{question} — focusing on {topic}", topic_ctx, mode))
                meta  = topic_docs[0].metadata
                fname = str(meta.get("source", "?")).replace("\\", "/").split("/")[-1]
                page  = meta.get("page_label", meta.get("page", "?"))
                topics_out.append({
                    "name": topic.title(),
                    "meta": f"{fname} · Page {page}",
                    "body": answer_text,
                })
            return jsonify({"topics": topics_out})

        # ── Standard mode ──
        docs = db.similarity_search(question, k=6)

        if not docs:
            return jsonify({"error": "No relevant content found in your documents."}), 404

        context = "\n\n---\n\n".join([
            f"[{d.metadata.get('source','?').replace(chr(92),'/').split('/')[-1]} "
            f"| Page {d.metadata.get('page_label', d.metadata.get('page','?'))}]\n{d.page_content}"
            for d in docs
        ])

        answer = ask_gemma(api_key, build_prompt(question, context, mode))

        # Deduplicate sources
        seen, sources = set(), []
        for d in docs:
            meta   = d.metadata
            source = meta.get("source", "Unknown")
            page   = str(meta.get("page_label", meta.get("page", "N/A")))
            key    = f"{source}-{page}"
            if key in seen:
                continue
            seen.add(key)
            fname = source.replace("\\", "/").split("/")[-1]
            sources.append({"name": fname, "page": page})

        # Return actual chunk texts so frontend can display them
        chunk_texts = [d.page_content[:400] for d in docs]

        return jsonify({
            "answer":      answer,
            "sources":     sources,
            "chunk_texts": chunk_texts   # ← NEW: real chunk content
        })

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
