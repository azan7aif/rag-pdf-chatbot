"""Streamlit UI for the RAG PDF chatbot.  Run:  streamlit run app.py"""
import os

import streamlit as st

# On Streamlit Community Cloud, keys live in "Secrets". Copy them into environment
# variables so rag.py (which reads os.getenv) works the same locally and online.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_key, _value)
except Exception:
    pass  # no secrets file locally: .env is used instead

import rag  # noqa: E402  (must come after the secrets bridge above)

SAMPLE_PDF = "sample_dbms_notes.pdf"

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄")
st.title("📄 Chat with your PDFs")
st.caption("Upload PDFs, then ask questions. Answers come only from your documents.")

if os.path.exists(SAMPLE_PDF) and not rag.list_documents():
    with st.spinner("Loading the sample document (first start only)..."):
        rag.ingest_pdf(SAMPLE_PDF, SAMPLE_PDF)


def show_sources(sources):
    if not sources:
        return
    with st.expander("Sources"):
        for s in sources:
            st.markdown(
                f"**{s['source']}** — page {s['page']}  \n"
                f"<small>distance: {s['distance']:.2f} (lower = more similar)</small>",
                unsafe_allow_html=True,
            )
            st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))


# ---------------- Sidebar: document management ----------------
with st.sidebar:
    st.header("Documents")
    st.caption("Demo app: please do not upload private or confidential documents.")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Process PDFs", disabled=not files):
        for f in files:
            with st.spinner(f"Processing {f.name}..."):
                n = rag.ingest_pdf(f, f.name)
            if n:
                st.success(f"{f.name}: {n} chunks stored")
            else:
                st.warning(f"{f.name}: no text found (scanned PDF? needs OCR)")

    docs = rag.list_documents()
    st.divider()
    if docs:
        st.write("**Indexed documents**")
        for d in docs:
            st.write(f"• {d}")
        to_delete = st.selectbox("Remove a document", ["-"] + docs)
        if to_delete != "-" and st.button("Delete"):
            rag.delete_document(to_delete)
            st.rerun()
    else:
        st.info("No documents yet.")

    scope = st.selectbox("Search in", ["All documents"] + docs)
    auto_k = st.checkbox("Auto-select number of chunks", value=True)
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, rag.TOP_K, disabled=auto_k)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Chat ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("sources"))

if question := st.chat_input("Ask something about your documents"):
    with st.chat_message("user"):
        st.markdown(question)

    history = st.session_state.messages[-6:]  # last few turns for follow-up questions
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply, sources = rag.answer(
                    question,
                    history=history,
                    k=None if auto_k else top_k,
                    source=None if scope == "All documents" else scope,
                )
            except Exception as e:  # e.g. missing API key
                reply, sources = f"Error: {e}", []
        st.markdown(reply)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
