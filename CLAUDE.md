# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

AWS credentials must be configured with access to Amazon Bedrock (Titan Embeddings + Claude models) in `us-west-2`:
```bash
aws configure
# or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION env vars
```

## Commands

**Ingest documents into the vector store (required before first query):**
```bash
python ingest.py
```

**Run demo queries against the knowledge base:**
```bash
python main.py
```

**Skip ingestion and query an existing vector store:**
```bash
python main.py --query-only
```

**Start an interactive Q&A session:**
```bash
python main.py --interactive
```

**Run the RAG chain directly (single question):**
```bash
python rag_chain.py
```

## Architecture

The pipeline has two phases that must run in order:

```
[documents/]  →  ingest.py  →  [vectorstore/]  →  rag_chain.py  →  answer
```

**Ingestion (`ingest.py`):**
Loads all `.txt` and `.pdf` files from `documents/`, splits them into overlapping chunks (`chunk_size=1000`, `chunk_overlap=200`), generates embeddings via Amazon Titan (`amazon.titan-embed-text-v2:0`), and saves the FAISS index to `vectorstore/` on disk.

**RAG Chain (`rag_chain.py`):**
Loads the persisted FAISS index, wraps it as a retriever (top-3 chunks by cosine similarity), and builds an LCEL chain:
```
question → retriever → format_docs → RAG_PROMPT → ChatBedrockConverse → StrOutputParser → answer
```
The prompt instructs the LLM to answer only from the retrieved context and to decline if the answer is not present.

**Entry point (`main.py`):**
Orchestrates ingestion + querying in sequence. Supports `--query-only` to skip ingestion and `--interactive` for a REPL session.

## Key Configuration

All model IDs and region are defined as module-level constants — change them in one place:

| Constant | File | Default |
|---|---|---|
| `EMBEDDING_MODEL_ID` | `ingest.py`, `rag_chain.py` | `amazon.titan-embed-text-v2:0` |
| `LLM_MODEL_ID` | `rag_chain.py` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| `AWS_REGION` | both | `us-west-2` |
| `TOP_K_RESULTS` | `rag_chain.py` | `3` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `ingest.py` | `1000` / `200` |

## Adding Documents

Drop `.txt` or `.pdf` files into `documents/` and re-run `python ingest.py` to rebuild the vector store. The existing `vectorstore/` directory is overwritten on each ingest run.
