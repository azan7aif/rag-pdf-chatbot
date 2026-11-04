"""
Core RAG pipeline:
PDF -> text -> chunks -> embeddings -> Chroma (vector DB) -> retrieve -> LLM answer
"""
import os
import time

import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------- Settings (can be changed from the .env file) ----------
EMBED_MODEL = "all-MiniLM-L6-v2"
DB_PATH = os.getenv("DB_PATH", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Auto mode: decide the number of chunks per question (instead of a fixed k)
AUTO_K = os.getenv("AUTO_K", "true").lower() == "true"
MAX_K = int(os.getenv("MAX_K", "8"))                   # look at most this many chunks
MIN_K = int(os.getenv("MIN_K", "1"))                   # always keep at least this many
RELEVANCE_GAP = float(os.getenv("RELEVANCE_GAP", "0.15"))  # keep chunks within this distance of the best one
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "1.0"))     # if even the best chunk is farther, say "not found"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",  # check Anthropic docs for the latest model names
    "gemini": "gemini-2.5-flash",      # check Google AI docs for the latest model names
}
LLM_MODEL = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(LLM_PROVIDER, "")

# If the main model is busy (503) or rate-limited (429), these are tried next.
# Override in .env:  LLM_FALLBACK_MODELS=model-a,model-b
FALLBACK_MODELS = [
    m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()
]
if not FALLBACK_MODELS and LLM_PROVIDER == "gemini":
    FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash"]

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the user's documents. "
    "Use ONLY the provided context. If the answer is not in the context, say: "
    "\"I couldn't find that in the uploaded documents.\" "
    "Do not make up facts. Mention the source file and page number when useful."
)

_embedder = None
_collection = None


# ---------- Loaders (created once, reused) ----------
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_PATH)
        _collection = client.get_or_create_collection(
            name="docs", metadata={"hnsw:space": "cosine"}
        )
    return _collection


# ---------- Step 1 & 2: read PDF and chunk it ----------
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks so context isn't cut in the middle."""
    text = " ".join(text.split())  # clean extra spaces/newlines
    if not text:
        return []
    chunks, start = [], 0
    step = max(size - overlap, 1)
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


# ---------- Step 3 & 4: embed and store ----------
def ingest_pdf(file, name):
    """
    file: path or file-like object (Streamlit upload works directly)
    name: display name, used as 'source' in metadata
    Returns the number of chunks stored (0 means no text found, e.g. scanned PDF).
    """
    reader = PdfReader(file)
    ids, docs, metas = [], [], []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for j, chunk in enumerate(chunk_text(text)):
            ids.append(f"{name}::p{page_no}::c{j}")
            docs.append(chunk)
            metas.append({"source": name, "page": page_no})

    if not docs:
        return 0

    collection = get_collection()
    batch = 256
    for i in range(0, len(docs), batch):
        part = docs[i:i + batch]
        embeddings = get_embedder().encode(part, show_progress_bar=False).tolist()
        collection.upsert(
            ids=ids[i:i + batch],
            documents=part,
            embeddings=embeddings,
            metadatas=metas[i:i + batch],
        )
    return len(docs)


def list_documents():
    data = get_collection().get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"]})


def delete_document(name):
    get_collection().delete(where={"source": name})


# ---------- Step 5: retrieve ----------
def select_relevant(chunks):
    """Auto mode: keep only chunks close in relevance to the best one (lower distance = better)."""
    if not chunks:
        return []
    best = chunks[0]["distance"]
    return [
        c
        for i, c in enumerate(chunks)
        if i < MIN_K or c["distance"] <= best + RELEVANCE_GAP
    ]


def retrieve(question, k=None, source=None):
    """
    k=None  -> auto mode (if AUTO_K is true): fetch up to MAX_K and keep only the relevant ones.
    k=<int> -> fixed number of chunks.
    """
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    auto = k is None and AUTO_K
    fetch = MAX_K if auto else (k or TOP_K)

    q_emb = get_embedder().encode(question).tolist()
    kwargs = {
        "query_embeddings": [q_emb],
        "n_results": min(fetch, total),
        "include": ["documents", "metadatas", "distances"],
    }
    if source:
        kwargs["where"] = {"source": source}

    res = collection.query(**kwargs)
    chunks = [
        {"text": d, "source": m["source"], "page": m["page"], "distance": dist}
        for d, m, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]
    return select_relevant(chunks) if auto else chunks


# ---------- Step 6: generate answer with an LLM ----------
def build_prompt(question, chunks, history=None):
    context = "\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}" for c in chunks
    )
    convo = ""
    if history:
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in history]
        convo = "Previous conversation:\n" + "\n".join(lines) + "\n\n"
    return f"{convo}Context:\n{context}\n\nQuestion: {question}"


def _call_llm(model, user_prompt):
    """One single call to the chosen LLM provider."""
    if LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        resp = client.messages.create(
            model=model,
            max_tokens=900,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text

    if LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        resp = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        return resp.text

    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'anthropic' or 'gemini'.")


def _is_temporary(error):
    """True for 'server busy' or 'rate limit' errors that are worth retrying."""
    text = str(error)
    return any(
        word in text
        for word in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded", "Overloaded")
    )


def generate(user_prompt, retries=3):
    """Call the LLM. On temporary errors: retry with waiting, then try fallback models."""
    models = [LLM_MODEL] + [m for m in FALLBACK_MODELS if m != LLM_MODEL]
    last_error = None
    for model in models:
        for attempt in range(retries):
            try:
                return _call_llm(model, user_prompt)
            except Exception as e:
                last_error = e
                if not _is_temporary(e):
                    raise  # real problem (wrong key, wrong model name...), show it
                time.sleep(2 * (2 ** attempt))  # wait 2s, 4s, 8s
        print(f"[warn] model '{model}' is busy, trying the next one...")
    raise last_error


def answer(question, history=None, k=None, source=None):
    """Full RAG: retrieve relevant chunks, then ask the LLM. Returns (answer, chunks)."""
    chunks = retrieve(question, k=k, source=source)
    if not chunks:
        return "No documents found yet. Please upload and process a PDF first.", []

    auto = k is None and AUTO_K
    if auto and chunks[0]["distance"] > MAX_DISTANCE:
        # Even the best match is too far: skip the LLM call
        return "I couldn't find that in the uploaded documents.", []

    return generate(build_prompt(question, chunks, history)), chunks


if __name__ == "__main__":
    # Quick terminal test:  python rag.py your_file.pdf "your question"
    import sys

    if len(sys.argv) >= 3:
        n = ingest_pdf(sys.argv[1], os.path.basename(sys.argv[1]))
        print(f"Stored {n} chunks.")
        reply, srcs = answer(sys.argv[2])
        print("\nAnswer:\n", reply)
        print("\nSources:", [(s["source"], s["page"]) for s in srcs])
    else:
        print('Usage: python rag.py file.pdf "your question"')
