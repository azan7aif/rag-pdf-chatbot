# 📄 Chat with your PDFs (RAG Chatbot)

Upload PDFs (college notes, company reports) and ask questions in natural language.
Answers are generated **only from your documents**, with source file and page number shown.

## How it works
PDF → text → overlapping chunks → embeddings (`all-MiniLM-L6-v2`) → ChromaDB →
top-k retrieval for each question → LLM answers from the retrieved context.

## Tech stack
Python · Sentence-Transformers · ChromaDB · Streamlit · Claude API / Gemini API

## Setup
```bash
git clone <your-repo-url>
cd rag-pdf-chatbot
python -m venv venv
venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env       # Mac/Linux: cp .env.example .env  -> then add your API key
streamlit run app.py
```

## Evaluation
`python evaluate.py` measures retrieval hit rate on my own test questions.

| Chunk size | k=1 | k=3 | k=5 |
|-----------|-----|-----|-----|
| 500       | __% | __% | __% |
| 800       | __% | __% | __% |
| 1200      | __% | __% | __% |

(Fill in with your real results.)

## Limitations
- Scanned PDFs need OCR (not included).
- Follow-up questions use recent chat history, but retrieval uses only the latest question.

## Future work
Hybrid search (BM25 + vectors), re-ranking, OCR support, streaming responses.
