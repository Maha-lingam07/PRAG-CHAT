# PRAG

**A Self-Healing RAG system**

PRAG is a PDF chat app with a small FastAPI backend. Python extracts PDF text, creates NVIDIA embeddings, stores and retrieves passages in Pinecone, and asks NVIDIA to answer with source page references.

## Run from Windows Command Prompt

In the project folder, run:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
copy .env.example .env
notepad .env
```

Put your real keys in `.env` and save it:

```dotenv
NVIDIA_API_KEY=your_nvidia_key
PINECONE_API_KEY=your_pinecone_key
```

Back in Command Prompt, start the server:

```bat
python -m uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Choose a PDF, enter a vector database name, and click **Upload PDF**. To connect to an existing index, enter its name in the lower field and click **Load database**. The gear icon shows whether the backend found both keys in `.env` and lets you set a Pinecone namespace.

The app creates a new index with 1,024 dimensions and cosine similarity in AWS `us-east-1` by default. It uses the index name as the Pinecone namespace unless you set a custom one. `.env` is ignored by Git, and the server only serves `index.html`, `app.js`, and `styles.css`; it does not serve `.env` or Python files.

Scanned PDFs need OCR and are not supported yet.

## Palette

Marigold `#FFB900`, pale yellow `#FFF48F`, sage `#496F39`, and forest `#243B23`.
