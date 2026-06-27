"""
PaperMind — Streamlit chat UI.

Thin presentation layer: handles PDF uploads and the chat, forwarding questions
to the backend. All RAG logic lives in backend/rag.py.

Run from the project root:
    streamlit run frontend/app.py
"""

import os
import sys
import uuid
import tempfile

import streamlit as st
from dotenv import load_dotenv

# Add the project root to the import path so `from backend.rag import ...` works
# regardless of where streamlit is launched from.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

# Resolve the Groq key before importing the backend, which builds the LLM client
# at import time. Prefer Streamlit Cloud secrets (set in the dashboard); fall back
# to a local .env. Reading st.secrets raises when no secrets file exists, so guard
# it for local dev.
load_dotenv(os.path.join(PROJECT_DIR, ".env"))
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY is not set. On Streamlit Cloud, add it under "
        "**Settings → Secrets**; for local dev, put it in a `.env` file."
    )
    st.stop()

from backend.rag import get_response, ingest_pdf, has_documents   # noqa: E402


st.set_page_config(page_title="PaperMind", page_icon="📄")

st.title("📄 PaperMind")
st.caption("Upload your PDFs and ask questions — answers are grounded in your documents, with citations.")


# Per-session state: a unique thread id (isolates conversation memory) and the
# set of filenames already ingested this session (so reruns don't re-embed them).
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested" not in st.session_state:
    st.session_state.ingested = []


# ── SIDEBAR: upload + library ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Your documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # Ingest any newly uploaded files. file_uploader re-returns the same files on
    # every rerun, so we skip ones we've already embedded this session.
    for uploaded in uploaded_files or []:
        if uploaded.name in st.session_state.ingested:
            continue
        with st.spinner(f"Embedding {uploaded.name}..."):
            # PyPDFLoader needs a path, so write the upload to a temp file first.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            n_chunks = ingest_pdf(tmp_path, uploaded.name)
            os.unlink(tmp_path)
        st.session_state.ingested.append(uploaded.name)
        st.success(f"Added {uploaded.name} ({n_chunks} chunks)")

    st.divider()

    # Library = documents uploaded this session.
    if st.session_state.ingested:
        st.subheader("Library")
        for name in st.session_state.ingested:
            st.markdown(f"- {name}")
    else:
        st.info("Upload a PDF above to begin, then ask questions about it in the chat.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# ── CHAT TRANSCRIPT ───────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── INPUT + RESPONSE ──────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not has_documents():
            # Nothing to retrieve from yet — nudge instead of returning "I don't know".
            answer = "Your library is empty. Upload a PDF in the sidebar to get started, then ask away."
            st.markdown(answer)
        else:
            with st.spinner("Searching your documents..."):
                answer = get_response(question, thread_id=st.session_state.thread_id)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
