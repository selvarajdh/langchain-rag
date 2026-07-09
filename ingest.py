"""
Document ingestion pipeline: load → split → embed → save FAISS vector store.
Run this once (or whenever documents change) before querying.
"""

import os
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
AWS_REGION = "us-west-2"
VECTOR_STORE_PATH = "vectorstore"
DOCS_PATH = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents(docs_path: str = DOCS_PATH):
    documents = []

    txt_loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
    documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks, store_path: str = VECTOR_STORE_PATH):
    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        region_name=AWS_REGION,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(store_path)
    print(f"Saved vector store to '{store_path}' ({len(chunks)} chunks from {len(set(c.metadata.get('source', '') for c in chunks))} source(s)).")
    return vector_store


if __name__ == "__main__":
    print(f"Loading documents from '{DOCS_PATH}'...")
    docs = load_documents()
    if not docs:
        print(f"No documents found in '{DOCS_PATH}'. Add .txt or .pdf files and re-run.")
        raise SystemExit(1)
    print(f"Loaded {len(docs)} document(s).")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    create_vector_store(chunks)
