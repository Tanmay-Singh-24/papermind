# 📄 PaperMind

**Chat with your own PDFs.** Upload one or more documents and ask questions answered *only* from their content — with citations to the source document. Ask a follow-up like *"who wrote it?"* and PaperMind understands what "it" refers to. If an answer isn't in your documents, it says so instead of guessing.

A clean, self-contained Retrieval-Augmented Generation (RAG) app built with LangGraph.

---

## The problem it solves

PDFs are dense, long, and hard to search. Finding *"the one paragraph that explains X"* across a stack of documents means a lot of skimming. A plain chatbot will happily **hallucinate** details it doesn't actually know, and it can't tell you *where* a claim came from.

PaperMind fixes both with **RAG**: it retrieves the most relevant passages from *your* uploaded documents and forces the model to answer **only** from them — citing the source document each time.

---

## Tech stack

- **Python**
- **LangGraph** — orchestrates the rewrite → retrieve → generate flow as a stateful graph, with native conversation memory
- **LangChain** — PDF loading, splitting, and integrations
- **ChromaDB** — persistent vector store for the embedded document chunks
- **HuggingFace embeddings** — `all-MiniLM-L6-v2` (384-dim sentence embeddings)
- **Groq** — `llama-3.3-70b-versatile` for fast, grounded answer generation
- **Streamlit** — minimal chat UI with PDF upload

---

## How it works

When you upload a PDF, it's loaded, split into ~1000-character chunks (100-char overlap so facts aren't cut across boundaries), embedded, and added to a persistent ChromaDB collection. Each chunk is tagged with its **source filename** so answers can cite it. The store persists to disk, so your library survives restarts.

### How retrieval works

Each question runs through a 3-node LangGraph:

```
   question ──► [rewrite_query] ──► [retrieve] ──► [generate] ──► grounded, cited answer
```

1. **rewrite_query** — Rewrites the question into a *standalone* search query using the conversation history. This is what makes follow-ups work: a bare *"who wrote it?"* has no keywords for the vector store to match, so we first resolve it to *"who wrote <document>?"*. (On the first turn there's nothing to resolve, so this step is skipped.)
2. **retrieve** — Embeds the standalone query and pulls the **top-5** most semantically similar chunks from ChromaDB.
3. **generate** — Hands those chunks (plus the conversation so far) to the LLM, with a system prompt that says: answer only from the context, admit when you don't know, and cite the document.

Conversation memory is handled by LangGraph's **checkpointer**, keyed by a per-session `thread_id` — no manual history wiring.

> **Note:** all uploads currently share one collection (no per-user isolation). The production fix would be per-session collections keyed by thread id, so one user's documents aren't visible to another.

---

## Setup & run

**Prerequisites:** Python 3.10+, a free [Groq API key](https://console.groq.com).

```bash
# 1. Clone and enter the project
git clone <your-repo-url> papermind
cd papermind

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
cp .env.example .env
# then edit .env and set GROQ_API_KEY=your_key_here

# 5. Run
streamlit run frontend/app.py
```

Then **upload a PDF in the sidebar** and start asking questions. The `sample_pdfs/` folder contains 5 landmark AI papers (Transformer, BERT, GPT-3, InstructGPT, ReAct) if you want something to try immediately.

The first upload takes a few seconds to embed; after that, queries are fast.

---

## Deploy (Streamlit Community Cloud)

The app runs on the free [Streamlit Community Cloud](https://share.streamlit.io). Its disk is **ephemeral** — the vector store rebuilds from whatever PDFs are uploaded after each restart, so users start with an empty library and fill it themselves.

1. **Push to GitHub.** Commit your changes and push to a GitHub repo (public or private).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub** and select your repo and branch.
4. Set **Main file path** to `frontend/app.py`.
5. Open **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
   (You can also add this later under **Settings → Secrets**.)
6. Click **Deploy**. The first build is slow — it installs `sentence-transformers` (which pulls in PyTorch), so allow several minutes.

---

## Project structure

```
papermind/
├── sample_pdfs/         # example PDFs to try (5 landmark AI papers)
├── backend/
│   ├── __init__.py
│   └── rag.py           # RAG pipeline: ingest, rewrite, retrieve, generate, memory
├── frontend/
│   └── app.py           # Streamlit UI: PDF upload + chat (thin — logic is in the backend)
├── chroma_store/        # persistent vector store (created on first upload, gitignored)
├── .streamlit/
│   └── secrets.toml.example   # secrets format for Streamlit Cloud / local secrets.toml
├── .env.example         # copy to .env and add your Groq key
├── requirements.txt
└── README.md
```
