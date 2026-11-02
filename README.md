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

The server and health check do not install PyTorch or load BGE. The large ML dependencies are in `ml-requirements.txt` and install lazily the first time an upload or question needs embeddings. Merely loading an existing Pinecone index does not install or load them. Asking a question does need BGE to embed the query, so after a fresh deploy or restart the first question can trigger the install and model download too. The model produces normalized 1,024-dimensional vectors, matching the notebook's embedding setup. `EMBEDDING_DEVICE=auto` chooses CUDA when PyTorch detects an NVIDIA GPU, Apple MPS when available, and otherwise CPU. If you explicitly set `cuda` but CUDA is unavailable, the app falls back to CPU. Once loaded, the model stays in memory for that server process.

Back in Command Prompt, start the server:

```bat
python -m uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Choose a PDF, enter a vector database name, and click **Upload PDF**. To connect to an existing index, enter its name in the lower field and click **Load database**. The gear icon shows whether the backend found both keys in `.env` and lets you set a Pinecone namespace.

The app creates a new index with 1,024 dimensions and cosine similarity in AWS `us-east-1` by default. It uses the index name as the Pinecone namespace unless you set a custom one. Keep the embedding model consistent within each index. Existing indexes populated by the original notebook's BGE embeddings are compatible. If you populated an index with the earlier PRAG version's NVIDIA embeddings, create a fresh index and upload the PDF again before querying it with BGE. `.env` is ignored by Git, and the server only serves `index.html`, `app.js`, and `styles.css`; it does not serve `.env` or Python files.

Scanned PDFs need OCR and are not supported yet.

## Deploy on Render

This repository includes `render.yaml` for a Render web service. In Render, create a new Blueprint from the GitHub repository and set `NVIDIA_API_KEY` and `PINECONE_API_KEY` in the service environment. The Blueprint binds to Render's `$PORT` and exposes `/api/health` as its health check. Keep the keys in Render's environment settings; do not commit a `.env` file.

Choose an instance with enough memory and disk for PyTorch and BGE Large. Render's free web service has 512 MB RAM, which is not enough for this local embedding model. Render's documented web-service plans list CPU and memory resources and do not list GPU instances, so `auto` will use CPU there. You can still run the same app on a GPU-enabled host, where `auto` selects CUDA. Render free services also spin down after inactivity, so the first embedding request after a restart may need to install the ML stack and load BGE again.

## Palette

Marigold `#FFB900`, pale yellow `#FFF48F`, sage `#496F39`, and forest `#243B23`.
