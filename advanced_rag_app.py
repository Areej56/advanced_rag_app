# setx GROQ_API_KEY "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"
import os
import pickle
from io import BytesIO

import streamlit as st
from typing import List, Dict, Any

from PyPDF2 import PdfReader
from docx import Document

# local embeddings
from sentence_transformers import SentenceTransformer

# raw FAISS
import faiss
import numpy as np

# Groq LLM
from groq import Groq

# ---------------------------
# CONFIG
# ---------------------------
DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_indexes")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Advanced RAG (Groq + Local Embeddings)", layout="wide")
st.title("📄 Advanced RAG — Groq + Sentence-Transformers + FAISS (Local)")

# ---------------------------
# UTIL: text extraction
# ---------------------------
def extract_text_from_pdf(file_bytes: BytesIO) -> str:
    reader = PdfReader(file_bytes)
    pages = []
    for p in reader.pages:
        try:
            t = p.extract_text()
        except Exception:
            t = ""
        if t:
            pages.append(t)
    return "\n\n".join(pages)

def extract_text_from_docx(file_bytes: BytesIO) -> str:
    doc = Document(file_bytes)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)

def extract_text_from_txt(file_bytes: BytesIO) -> str:
    raw = file_bytes.read()
    try:
        return raw.decode("utf-8")
    except Exception:
        return raw.decode("latin-1", errors="ignore")

# ---------------------------
# MODEL: cache SentenceTransformer
# ---------------------------
@st.cache_resource
def load_st_model(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

st.sidebar.markdown("### Index / Model settings")
model_name = st.sidebar.text_input("SentenceTransformers model", value="all-MiniLM-L6-v2")
st.sidebar.write("Model will download on first run.")

emb_model = load_st_model(model_name)

# ---------------------------
# FAISS helpers
# ---------------------------
def build_faiss_index(vectors: np.ndarray):
    """Create an IndexFlatL2 and add vectors (vectors: (N, D) float32). Returns the index."""
    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    return index

def save_faiss_index(index: faiss.Index, index_name: str, texts: List[str], metadatas: List[Dict[str, Any]]):
    idx_path = os.path.join(INDEX_DIR, f"{index_name}.index")
    meta_path = os.path.join(INDEX_DIR, f"{index_name}_meta.pkl")
    faiss.write_index(index, idx_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)
    return idx_path, meta_path

def load_faiss_index(index_name: str):
    idx_path = os.path.join(INDEX_DIR, f"{index_name}.index")
    meta_path = os.path.join(INDEX_DIR, f"{index_name}_meta.pkl")
    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        return None, None, None
    index = faiss.read_index(idx_path)
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    return index, data["texts"], data["metadatas"]

def search_index(index: faiss.Index, query_vec: np.ndarray, k: int = 4):
    """query_vec shape (1, D) float32 -> returns (distances, indices)"""
    distances, indices = index.search(query_vec, k)
    return distances[0], indices[0]

# ---------------------------
# GROQ helper
# ---------------------------
def groq_chat(prompt: str, temperature: float = 0.2) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment.")
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama3-8b-2048",  # smaller by default; change if you have resources
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    # robustly extract text
    try:
        return completion.choices[0].message["content"]
    except Exception:
        # fallback to string representation
        return str(completion)

# ---------------------------
# UI: sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Settings")
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=200, step=50)
    top_k = st.number_input("Top K retrieved", min_value=1, max_value=10, value=4)
    llm_temperature = st.slider("LLM temperature", 0.0, 1.0, 0.1)
    index_name = st.text_input("Index name", value="default_index")

# ---------------------------
# Upload files
# ---------------------------
st.subheader("1) Upload documents (PDF, DOCX, TXT)")

uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

all_texts_pairs = []  # list of (text, filename)
if uploaded_files:
    for f in uploaded_files:
        buf = BytesIO(f.getbuffer())
        if f.name.lower().endswith(".pdf"):
            txt = extract_text_from_pdf(buf)
        elif f.name.lower().endswith(".docx"):
            txt = extract_text_from_docx(buf)
        else:
            txt = extract_text_from_txt(buf)
        if txt and txt.strip():
            all_texts_pairs.append((txt, f.name))
            # save uploaded file for provenance
            save_path = os.path.join(UPLOAD_DIR, f.name)
            with open(save_path, "wb") as wf:
                wf.write(f.getbuffer())
    st.success(f"Loaded {len(all_texts_pairs)} file(s)")

# ---------------------------
# Chunking
# ---------------------------
st.subheader("2) Split into chunks & Build index")

if all_texts_pairs:
    if st.button("Prepare chunks"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = []
        metadatas = []
        for text, fname in all_texts_pairs:
            chunks = splitter.split_text(text)
            for i, c in enumerate(chunks):
                texts.append(c)
                metadatas.append({"source": fname, "chunk": i})
        st.session_state["texts"] = texts
        st.session_state["metadatas"] = metadatas
        st.success(f"Prepared {len(texts)} chunks")

# ---------------------------
# Build embeddings & FAISS index
# ---------------------------
if "texts" in st.session_state and st.session_state["texts"]:
    if st.button("Build FAISS index"):
        texts = st.session_state["texts"]
        metadatas = st.session_state["metadatas"]

        with st.spinner("Computing embeddings (local) ..."):
            # produce numpy float32 vectors
            vectors = emb_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            vectors = np.array(vectors).astype("float32")

        with st.spinner("Building FAISS index ..."):
            index = build_faiss_index(vectors)
            save_faiss_index(index, index_name, texts, metadatas)

        st.session_state["faiss_index_name"] = index_name
        st.success("FAISS index built and saved.")

# ---------------------------
# Load existing index
# ---------------------------
if st.button("Load persisted index"):
    idx, texts_loaded, metas_loaded = load_faiss_index(index_name)
    if idx is None:
        st.error("No persisted index found under this name.")
    else:
        st.session_state["loaded_index"] = idx
        st.session_state["loaded_texts"] = texts_loaded
        st.session_state["loaded_metas"] = metas_loaded
        st.success("Loaded persisted index into session.")

# ---------------------------
# Query / RAG
# ---------------------------
st.subheader("3) Ask questions (RAG)")

query = st.text_input("Enter your question here")
if st.button("Ask") and query:
    # prefer loaded index; else use recently built one
    index = st.session_state.get("loaded_index")
    if index is None:
        # try to load the built index from disk
        idx, texts_list, metas_list = load_faiss_index(st.session_state.get("faiss_index_name", index_name))
        if idx is None:
            st.error("No index available — build or load one first.")
            st.stop()
        index = idx
        texts_list = texts_list
        metas_list = metas_list
    else:
        texts_list = st.session_state.get("loaded_texts")
        metas_list = st.session_state.get("loaded_metas")

    # encode query
    q_vec = emb_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_vec, top_k)
    retrieved = []
    for idx_hit in I[0]:
        if idx_hit < 0 or idx_hit >= len(texts_list):
            continue
        retrieved.append({"text": texts_list[idx_hit], "metadata": metas_list[idx_hit]})

    # Build context prompt with provenance
    context = ""
    for r in retrieved:
        src = r["metadata"].get("source", "unknown")
        chunk = r["metadata"].get("chunk", "")
        context += f"\nSOURCE: {src} (chunk {chunk})\n{r['text']}\n\n"

    prompt = f"""You are an assistant. Use ONLY the following context to answer the question. If the answer is not in the context, say you don't know.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    try:
        with st.spinner("Generating answer via Groq..."):
            answer = groq_chat(prompt, temperature=llm_temperature)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Groq generation failed: {e}")
        st.stop()

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved sources")
    for r in retrieved:
        st.markdown(f"**{r['metadata'].get('source','unknown')}** — chunk {r['metadata'].get('chunk','')}")
        st.code(r['text'][:800] + ("..." if len(r['text'])>800 else ""))

    # save history
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({"query": query, "answer": answer, "retrieved": retrieved})

# ---------------------------
# Show chat history
# ---------------------------
if "history" in st.session_state and st.session_state["history"]:
    st.markdown("---")
    st.subheader("Previous Q/A")
    for turn in reversed(st.session_state["history"]):
        st.markdown(f"**Q:** {turn['query']}")
        st.markdown(f"**A:** {turn['answer'][:1000]}{'...' if len(turn['answer'])>1000 else ''}")
        st.markdown("---")
