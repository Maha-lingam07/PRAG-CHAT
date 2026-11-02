# PRAG

**A Self-Healing RAG system**

PRAG is a PDF chat app with a lightweight FastAPI backend. Python extracts PDF text, creates NVIDIA embeddings, stores and retrieves passages in Pinecone, then asks NVIDIA to answer from the retrieved context with source pages.

## Run locally

Use Python 3.10 or newer. From the project folder, create a virtual environment and install dependencies:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your NVIDIA and Pinecone keys to `.env`:

```dotenv
NVIDIA_API_KEY=your_nvidia_key
PINECONE_API_KEY=your_pinecone_key
```

Start the app and open [http://127.0.0.1:8000](http://127.0.0.1:8000):

```powershell
uvicorn main:app --reload
```

Choose a PDF, enter a vector database name, and click **Upload PDF**. PRAG creates the Pinecone index when needed (1,024 dimensions, cosine metric, AWS `us-east-1` by default) and uploads extracted passages. Enter an existing index in the lower field and click **Load database** to connect. The gear icon shows backend configuration and lets you set a Pinecone namespace; if blank, the index name is used as the namespace.

The `.env` file is ignored by Git. The server only serves the app's HTML, JavaScript, and CSS files, so `.env` and Python files are not exposed as static files. API keys stay on the Python server and are never returned to the browser. Scanned PDFs need OCR and are not supported yet.

## Palette

Marigold `#FFB900`, pale yellow `#FFF48F`, sage `#496F39`, and forest `#243B23`.
