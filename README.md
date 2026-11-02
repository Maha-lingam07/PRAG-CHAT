# PRAG

**A Self-Healing RAG system**

PRAG is a PDF chat app with a small FastAPI backend. Python extracts PDF text, embeds it with `BAAI/bge-large-en-v1.5`, stores and retrieves passages in Pinecone, and uses NVIDIA Nemotron for answers with source page references. This matches the models used in the original notebook.

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

The first PDF upload or question downloads and initializes the BGE model on the backend. The model produces normalized 1,024-dimensional vectors, matching the notebook's embedding setup. Set `EMBEDDING_DEVICE=cuda` in `.env` to request CUDA; `auto` uses the available default device. The BGE model and its PyTorch runtime are larger than the FastAPI server itself, so the first dependency install and model download take longer.

Back in Command Prompt, start the server:

```bat
python -m uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Choose a PDF, enter a vector database name, and click **Upload PDF**. To connect to an existing index, enter its name in the lower field and click **Load database**. The gear icon shows whether the backend found both keys in `.env` and lets you set a Pinecone namespace.

The app creates a new index with 1,024 dimensions and cosine similarity in AWS `us-east-1` by default. It uses the index name as the Pinecone namespace unless you set a custom one. Keep the embedding model consistent within each index. Existing indexes populated by the original notebook's BGE embeddings are compatible. If you populated an index with the earlier PRAG version's NVIDIA embeddings, create a fresh index and upload the PDF again before querying it with BGE. `.env` is ignored by Git, and the server only serves `index.html`, `app.js`, and `styles.css`; it does not serve `.env` or Python files.

Scanned PDFs need OCR and are not supported yet.

## Palette

Marigold `#FFB900`, pale yellow `#FFF48F`, sage `#496F39`, and forest `#243B23`.
