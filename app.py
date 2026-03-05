import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai

DB_PATH = "embeddings"

st.set_page_config(
    page_title="CYNOTE — AI Study Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#05070f;--bg2:#090d1a;--surface:#0e1325;--border:rgba(99,202,255,0.12);--accent:#63caff;--accent2:#a78bfa;--accent3:#34d399;--text:#e2e8f0;--muted:#64748b;}
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background-color:var(--bg)!important;color:var(--text)!important;}
.stApp{background:var(--bg)!important;background-image:radial-gradient(ellipse 80% 50% at 50% -20%,rgba(99,202,255,0.07) 0%,transparent 60%),radial-gradient(ellipse 60% 40% at 80% 100%,rgba(167,139,250,0.05) 0%,transparent 60%)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 3rem 4rem 3rem!important;max-width:1100px!important;}
section[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border)!important;padding-top:2rem!important;}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
.cynote-header{display:flex;align-items:center;gap:1.2rem;margin-bottom:0.4rem;}
.cynote-logo{width:52px;height:52px;background:linear-gradient(135deg,#63caff22,#a78bfa22);border:1px solid var(--border);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.6rem;box-shadow:0 0 24px rgba(99,202,255,0.15);}
.cynote-title{font-family:'Syne',sans-serif!important;font-size:2.6rem!important;font-weight:800!important;letter-spacing:-1px!important;background:linear-gradient(135deg,#63caff,#a78bfa)!important;-webkit-background-clip:text!important;-webkit-text-fill-color:transparent!important;background-clip:text!important;line-height:1!important;margin:0!important;}
.cynote-sub{font-size:0.8rem;letter-spacing:3px;text-transform:uppercase;color:var(--muted)!important;font-family:'DM Mono',monospace;margin-top:2px;}
.cynote-tagline{font-size:0.95rem;color:var(--muted)!important;margin-bottom:2.5rem;font-weight:300;}
.cynote-tagline span{color:var(--accent)!important;font-weight:500;}
.stat-row{display:flex;gap:0.75rem;margin-bottom:2.5rem;flex-wrap:wrap;}
.stat-pill{display:inline-flex;align-items:center;gap:0.4rem;padding:0.35rem 0.9rem;border-radius:999px;font-size:0.75rem;font-family:'DM Mono',monospace;font-weight:500;letter-spacing:0.5px;border:1px solid;}
.pill-blue{background:rgba(99,202,255,0.06);border-color:rgba(99,202,255,0.25);color:#63caff!important;}
.pill-purple{background:rgba(167,139,250,0.06);border-color:rgba(167,139,250,0.25);color:#a78bfa!important;}
.pill-green{background:rgba(52,211,153,0.06);border-color:rgba(52,211,153,0.25);color:#34d399!important;}
.stTextInput>div>div>input{background:var(--surface)!important;border:1px solid rgba(99,202,255,0.2)!important;border-radius:14px!important;color:var(--text)!important;font-size:1rem!important;padding:1rem 1.25rem!important;font-family:'Inter',sans-serif!important;transition:all 0.2s ease!important;}
.stTextInput>div>div>input:focus{border-color:rgba(99,202,255,0.55)!important;box-shadow:0 0 0 3px rgba(99,202,255,0.08)!important;outline:none!important;}
.stTextInput>div>div>input::placeholder{color:var(--muted)!important;}
.stTextInput label{font-family:'Syne',sans-serif!important;font-weight:600!important;font-size:0.8rem!important;letter-spacing:1.5px!important;text-transform:uppercase!important;color:var(--muted)!important;}
.stButton>button{background:linear-gradient(135deg,#1a3a52,#1e2d4d)!important;color:var(--accent)!important;border:1px solid rgba(99,202,255,0.35)!important;border-radius:12px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:0.9rem!important;letter-spacing:1.5px!important;text-transform:uppercase!important;padding:0.7rem 2.5rem!important;transition:all 0.2s ease!important;box-shadow:0 4px 20px rgba(99,202,255,0.1)!important;}
.stButton>button:hover{border-color:var(--accent)!important;box-shadow:0 0 30px rgba(99,202,255,0.25)!important;transform:translateY(-1px)!important;color:#fff!important;}
.answer-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.75rem 2rem;margin:1.5rem 0;position:relative;overflow:hidden;}
.answer-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#63caff,#a78bfa,#34d399);}
.answer-label{font-family:'DM Mono',monospace;font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:var(--accent)!important;margin-bottom:0.75rem;font-weight:500;}
.answer-text{font-size:1rem;line-height:1.85;color:var(--text)!important;font-weight:300;white-space:pre-wrap;}
.source-card{background:rgba(99,202,255,0.03);border:1px solid rgba(99,202,255,0.1);border-radius:10px;padding:0.6rem 1rem;margin:0.4rem 0;display:flex;align-items:center;gap:0.75rem;font-family:'DM Mono',monospace;font-size:0.78rem;color:var(--muted)!important;}
.source-dot{width:6px;height:6px;border-radius:50%;background:var(--accent3);flex-shrink:0;}
.topic-card{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent2);border-radius:12px;padding:1.25rem 1.5rem;margin:1rem 0;}
.topic-title{font-family:'Syne',sans-serif;font-weight:700;font-size:0.85rem;letter-spacing:1.5px;text-transform:uppercase;color:var(--accent2)!important;margin-bottom:0.4rem;}
.topic-meta{font-family:'DM Mono',monospace;font-size:0.7rem;color:var(--muted)!important;margin-bottom:0.75rem;}
.topic-body{font-size:0.9rem;line-height:1.7;color:var(--text)!important;font-weight:300;white-space:pre-wrap;}
.sb-logo{font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;background:linear-gradient(135deg,#63caff,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.5px;margin-bottom:0.25rem;}
.sb-tagline{font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted)!important;margin-bottom:1.5rem;}
.sb-section-label{font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted)!important;margin:1.5rem 0 0.75rem 0;}
.sb-feature{display:flex;align-items:flex-start;gap:0.6rem;padding:0.5rem 0;font-size:0.82rem;color:#94a3b8!important;border-bottom:1px solid rgba(255,255,255,0.04);}
.sb-feature-icon{font-size:0.85rem;margin-top:1px;flex-shrink:0;}
div[data-testid="stRadio"] label{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:0.55rem 0.9rem!important;margin:0.2rem 0!important;font-size:0.82rem!important;color:var(--muted)!important;transition:all 0.15s ease!important;}
div[data-testid="stRadio"] label:hover{border-color:rgba(99,202,255,0.35)!important;color:var(--text)!important;}
hr{border:none!important;border-top:1px solid var(--border)!important;margin:2rem 0!important;}
.stAlert{background:rgba(99,202,255,0.05)!important;border:1px solid rgba(99,202,255,0.2)!important;border-radius:10px!important;}
::-webkit-scrollbar{width:6px;}::-webkit-scrollbar-track{background:var(--bg2);}::-webkit-scrollbar-thumb{background:#1e3a52;border-radius:3px;}::-webkit-scrollbar-thumb:hover{background:var(--accent);}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown('<div class="sb-logo">⚡ CYNOTE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tagline">AI Study Engine v2.0</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section-label">🔑 Google AI Studio Key</div>', unsafe_allow_html=True)
    api_key = st.text_input("", type="password", placeholder="AIza...", label_visibility="collapsed")
    if not api_key:
        st.markdown('<div style="font-family:monospace;font-size:0.7rem;color:#475569;margin-top:0.4rem;">→ aistudio.google.com/apikey  (FREE)</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section-label">Capabilities</div>', unsafe_allow_html=True)
    for icon, text in [
        ("🧠", "Gemma 3 27B — Google open model"),
        ("🤗", "Hosted free on HuggingFace"),
        ("📄", "Answers grounded in your notes"),
        ("🎯", "Exam-pattern optimised output"),
        ("🔒", "Your PDFs never leave your machine"),
    ]:
        st.markdown(f'<div class="sb-feature"><span class="sb-feature-icon">{icon}</span>{text}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section-label">Answer Mode</div>', unsafe_allow_html=True)
    mode = st.radio("", [
        "2 Marks — Brief",
        "5 Marks — Detailed",
        "10 Marks — In-depth",
        "Show All Related",
        "Build Exam Notes",
    ], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:monospace;font-size:0.65rem;color:#1e293b;letter-spacing:1px;">LANGCHAIN · FAISS · GEMMA 3 27B</div>', unsafe_allow_html=True)

# ── HERO ──
st.markdown("""
<div class="cynote-header">
  <div class="cynote-logo">⚡</div>
  <div>
    <div class="cynote-title">CYNOTE</div>
    <div class="cynote-sub">AI Study Engine</div>
  </div>
</div>
<p class="cynote-tagline">Ask anything from your notes. Get <span>exam-ready answers</span> via Gemma 3 27B.</p>
<div class="stat-row">
  <span class="stat-pill pill-blue">🧠 Gemma 3 27B</span>
  <span class="stat-pill pill-purple">🤗 HuggingFace — Free</span>
  <span class="stat-pill pill-green">✓ Notes-Grounded</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

question = st.text_input("YOUR QUESTION", placeholder="e.g. Explain software validation with examples...")
col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("⚡  ASK")

# ── PROMPT BUILDER ──
def build_prompt(question, context, mode):
    instructions = {
        "2 Marks — Brief":     "Write a concise 2-mark answer in 3-5 sentences. Be direct and factual.",
        "5 Marks — Detailed":  "Write a 5-mark answer with definition, explanation, and an example.",
        "10 Marks — In-depth": "Write a comprehensive 10-mark answer with intro, detailed sub-points, examples, and conclusion.",
        "Show All Related":    "List all relevant concepts from the notes with brief explanations for each.",
        "Build Exam Notes":    "Build structured exam revision notes with headings, key points, and important terms.",
    }
    return f"""You are CYNOTE, an expert exam study assistant. Answer ONLY using the provided notes.

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

# ── CALL GEMMA 3 27B via Google AI Studio ──
def ask_gemma(api_key, prompt):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=prompt,
    )
    return response.text

# ── ASK LOGIC ──
if ask:
    if not api_key:
        st.warning("⚠️ Add your HuggingFace token in the sidebar. Get it FREE at huggingface.co/settings/tokens")
        st.stop()
    if not question.strip():
        st.warning("Enter a question to get started.")
        st.stop()

    with st.spinner("Searching your notes..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(question, k=6)

    context = "\n\n---\n\n".join([
        f"[{d.metadata.get('source', '?').replace('\\\\', '/').split('/')[-1]} | Page {d.metadata.get('page_label', d.metadata.get('page', '?'))}]\n{d.page_content}"
        for d in docs
    ])

    if mode in ["10 Marks — In-depth", "Build Exam Notes"]:
        st.markdown('<div class="answer-label">📘 10-Mark Answer — Topic-wise</div>', unsafe_allow_html=True)
        topics = ["software specification", "software development", "software validation", "software evolution"]

        for topic in topics:
            topic_docs = db.similarity_search(f"{question} {topic}", k=4)
            topic_ctx  = "\n\n".join([d.page_content for d in topic_docs])
            with st.spinner(f"Writing: {topic}..."):
                answer_text = ask_gemma(api_key, build_prompt(f"{question} — focusing on {topic}", topic_ctx, mode))
            meta  = topic_docs[0].metadata if topic_docs else {}
            fname = str(meta.get("source", "?")).replace("\\\\", "/").split("/")[-1]
            page  = meta.get("page_label", meta.get("page", "?"))
            st.markdown(f'''<div class="topic-card">
  <div class="topic-title">◈ {topic.title()}</div>
  <div class="topic-meta">📄 {fname} · Page {page}</div>
  <div class="topic-body">{answer_text}</div>
</div>''', unsafe_allow_html=True)

    else:
        with st.spinner("Gemma 3 is thinking..."):
            answer = ask_gemma(api_key, build_prompt(question, context, mode))

        st.markdown(f'''<div class="answer-card">
  <div class="answer-label">✦ Answer — {mode}</div>
  <div class="answer-text">{answer}</div>
</div>''', unsafe_allow_html=True)

        st.markdown('<div class="answer-label" style="margin-top:1.5rem;">◈ Source References</div>', unsafe_allow_html=True)
        shown = set()
        for d in docs:
            meta   = d.metadata
            source = meta.get("source", "Unknown")
            page   = meta.get("page_label", meta.get("page", "N/A"))
            key    = f"{source}-{page}"
            if key in shown: continue
            shown.add(key)
            fname = source.replace("\\\\", "/").split("/")[-1]
            st.markdown(f'''<div class="source-card">
  <div class="source-dot"></div><span>{fname}</span>
  <span style="color:#1e293b">·</span><span>Page {page}</span>
</div>''', unsafe_allow_html=True)