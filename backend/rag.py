"""
PaperMind — RAG backend.

A "chat with your own PDFs" tool: the user uploads documents, they get embedded
into a persistent Chroma collection, and the user asks questions answered only
from those documents, with citations. Built with LangGraph.

Per-turn graph flow:

    question ─► [rewrite_query] ─► [retrieve] ─► [generate] ─► grounded, cited answer

The rewrite step resolves follow-up questions ("who wrote it?") into standalone
queries before retrieval. Memory is handled by LangGraph's checkpointer.
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


# ── PATHS ─────────────────────────────────────────────────────────────────────
# Resolved relative to this file so the app runs from any working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
CHROMA_DIR = os.path.join(PROJECT_DIR, "chroma_store")

# Load GROQ_API_KEY from a local .env at the project root (see .env.example).
load_dotenv(dotenv_path=os.path.join(PROJECT_DIR, ".env"))


# ── CONFIG ────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 5
COLLECTION_NAME = "papermind"


# Shared embedding model — the same instance embeds uploaded documents and the
# search query, so both live in the same vector space.
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# Connect to the persistent collection, creating an empty one if it doesn't
# exist. No documents are required to boot — the library starts empty and the
# user fills it by uploading PDFs.
#
# NOTE: all uploads share this single collection — there's no per-user isolation
# yet. The production fix would be per-session collections keyed by thread id, so
# one user's documents aren't visible to another.
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)


def ingest_pdf(file_path, source_name):
    """Embed a single PDF and ADD it to the collection (never rebuilds).

    `source_name` (the uploaded file's name) is stored on every chunk so answers
    can cite which document a fact came from. Returns the number of chunks added.
    """
    pages = PyPDFLoader(file_path).load()
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["source"] = source_name   # used for citations
    vectorstore.add_documents(chunks)
    return len(chunks)


def has_documents():
    """True if the collection holds at least one chunk (library is non-empty)."""
    return vectorstore._collection.count() > 0


# ── LLM ───────────────────────────────────────────────────────────────────────
# temperature=0: we want grounded, repeatable answers, not creativity.
llm = ChatGroq(model=LLM_MODEL, temperature=0)

SYSTEM_PROMPT = (
    "You are PaperMind, an assistant that answers questions grounded in the "
    "user's uploaded documents. Answer using ONLY the provided context from "
    "those documents. If the answer isn't in the context, say so honestly "
    "rather than guessing. When you answer, cite which document the information "
    "comes from."
)

# Used by rewrite_query. It must reformulate only — never answer — or it would
# pollute the search query with an answer the retriever can't match on.
REWRITE_PROMPT = (
    "Given the conversation so far and the latest user question, rewrite the "
    "question as a standalone search query that makes sense without the prior "
    "messages. Resolve references like 'it', 'they', or 'that document' to the "
    "actual subject from the conversation. If the question is already "
    "standalone, return it unchanged. Output ONLY the rewritten query — do not "
    "answer it."
)


# ── GRAPH STATE ───────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation (memory)
    search_query: str                         # standalone query for retrieval
    context: str                              # retrieved chunks for this turn


# ── NODE 1: REWRITE QUERY ─────────────────────────────────────────────────────
def rewrite_query(state: State) -> dict:
    """Turn the latest (possibly context-dependent) question into a standalone
    search query, so retrieval works on follow-ups like "who created it?".

    On the first turn there's no prior context to resolve, so we skip the LLM
    call and use the question as-is.
    """
    history = state["messages"]
    latest_question = history[-1].content

    if len(history) == 1:
        return {"search_query": latest_question}

    # Pass the whole history so the model can resolve pronouns from earlier turns.
    rewrite_messages = [SystemMessage(content=REWRITE_PROMPT)] + history
    rewritten = llm.invoke(rewrite_messages).content.strip()
    return {"search_query": rewritten}


# ── NODE 2: RETRIEVE ──────────────────────────────────────────────────────────
def retrieve(state: State) -> dict:
    """Fetch the top-K chunks most similar to the standalone search query."""
    docs = vectorstore.similarity_search(state["search_query"], k=TOP_K)
    blocks = [
        f"[Source: {doc.metadata.get('source', 'Unknown document')}]\n{doc.page_content}"
        for doc in docs
    ]
    return {"context": "\n\n---\n\n".join(blocks)}


# ── NODE 3: GENERATE ──────────────────────────────────────────────────────────
def generate(state: State) -> dict:
    """Answer from the retrieved context plus the conversation history."""
    system_content = f"{SYSTEM_PROMPT}\n\nContext from the documents:\n{state['context']}"
    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ── BUILD + COMPILE THE GRAPH ─────────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "rewrite_query")
builder.add_edge("rewrite_query", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

# Checkpointer persists each thread's state between calls, keyed by thread_id.
graph = builder.compile(checkpointer=InMemorySaver())


# ── PUBLIC API ────────────────────────────────────────────────────────────────
def get_response(question, thread_id):
    """Answer a question, remembering everything said under this thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=question)]}, config)
    return result["messages"][-1].content
