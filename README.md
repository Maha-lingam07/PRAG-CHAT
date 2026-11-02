# PRAG

**A Self-Healing RAG system**

PRAG is a browser-based PDF chat app. It extracts PDF text locally in the browser, creates 1,024-dimensional NVIDIA embeddings, stores passages in Pinecone, and answers questions with source page references.

## Run locally

From the project folder, start a local static server:

```sh
python -m http.server 8000
```

Open [http://localhost:8000](http://localhost:8000). Select the settings icon and enter NVIDIA and Pinecone API keys. To add a PDF, choose the file, enter a vector database name, and click **Upload PDF**. To connect to an existing index, enter its name in the lower field and click **Load database**.

PDF upload creates the Pinecone index if needed (1,024 dimensions, cosine metric, AWS `us-east-1`) and uses the index name as its namespace by default. Set a custom namespace in API settings when connecting to an index with a different namespace. Scanned PDFs need OCR and are not supported yet.

Keys stay in JavaScript memory in the current tab and are cleared on reload. The app makes direct browser requests to Pinecone and NVIDIA, so their CORS settings must allow requests from the page's origin.

## Palette

Marigold `#FFB900`, pale yellow `#FFF48F`, sage `#496F39`, and forest `#243B23`.
