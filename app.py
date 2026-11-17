import os
import time
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
# Change via GEMINI_MODEL in .env if this model name is retired
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TOP_K = 4


@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def get_llm():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def build_index(collection):
    """Read every PDF in docs/, split into chunks, embed, store in Chroma."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    embedder = get_embedder()
    ids, texts, metas = [], [], []

    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        reader = PdfReader(str(pdf))
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for i, chunk in enumerate(splitter.split_text(text)):
                ids.append(f"{pdf.stem}-p{page_no}-c{i}")
                texts.append(chunk)
                metas.append({"source": pdf.name, "page": page_no})

    for start in range(0, len(texts), 256):
        end = start + 256
        embeddings = embedder.encode(texts[start:end], show_progress_bar=False).tolist()
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings,
            metadatas=metas[start:end],
        )


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        "papers", metadata={"hnsw:space": "cosine"}
    )
    if collection.count() == 0:
        with st.spinner("Indexing PDFs (first run only)..."):
            build_index(collection)
    return collection


def retrieve(question):
    q_emb = get_embedder().encode([question]).tolist()
    res = get_collection().query(query_embeddings=q_emb, n_results=TOP_K)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def answer(question):
    hits = retrieve(question)
    context = "\n\n".join(
        f"[{m['source']}, page {m['page']}]\n{doc}" for doc, m in hits
    )
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know. "
        "Be concise.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    models_to_try = [LLM_MODEL, "gemini-3.8-flash-lite"]
    last_err = None

    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = get_llm().models.generate_content(
                    model=model_name, contents=prompt
                )
                return response.text, hits
            except Exception as e:
                last_err = e
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(2)
                    continue
                raise e

    raise last_err


st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("📄 Chat with AI Research Papers")
st.caption(f"Embeddings: {EMBED_MODEL} | Vector store: Chroma | LLM: {LLM_MODEL}")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found. Add it to your .env file.")
    st.stop()

if "messages" not in st.session_state: 
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about the papers..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply, hits = answer(question)
            except Exception as e:
                reply, hits = f"Error: {e}", []
        st.markdown(reply)
        if hits:
            with st.expander("Sources"):
                for doc, m in hits:
                    st.markdown(f"**{m['source']}, page {m['page']}**")
                    st.caption(doc[:300] + "...")
    st.session_state.messages.append({"role": "assistant", "content": reply})
