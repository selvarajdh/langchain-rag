"""
End-to-end RAG demo: ingest documents then run a question-answering session.
Usage:
  python main.py              # ingest + demo queries
  python main.py --query-only # skip ingestion, query existing vector store
"""

import argparse
import os

from ingest import create_vector_store, load_documents, split_documents, VECTOR_STORE_PATH
from rag_chain import build_rag_chain

DEMO_QUESTIONS = [
    "What is Retrieval-Augmented Generation and how does it work?",
    "What is FAISS and why is it used in RAG pipelines?",
    "What models does Amazon Bedrock support?",
    "What is Claude Code and what are its key features?",
    "How does LangChain Expression Language (LCEL) work?",
]


def ingest():
    print("=== Ingestion ===")
    docs = load_documents()
    if not docs:
        raise FileNotFoundError(
            "No documents found in 'documents/'. Add .txt or .pdf files first."
        )
    print(f"Loaded {len(docs)} document(s).")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    create_vector_store(chunks)
    print()


def run_demo_queries(chain):
    print("=== RAG Query Session ===\n")
    for question in DEMO_QUESTIONS:
        print(f"Q: {question}")
        answer = chain.invoke(question)
        print(f"A: {answer}")
        print("-" * 60)


def interactive_query(chain):
    print("=== Interactive Mode (type 'exit' to quit) ===\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue
        answer = chain.invoke(question)
        print(f"A: {answer}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain RAG with AWS Bedrock")
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Skip ingestion and query the existing vector store",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive Q&A session instead of running demo queries",
    )
    args = parser.parse_args()

    if not args.query_only:
        ingest()
    elif not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTOR_STORE_PATH}'. Run without --query-only first."
        )

    print("Loading RAG chain...")
    chain = build_rag_chain()
    print("Chain ready.\n")

    if args.interactive:
        interactive_query(chain)
    else:
        run_demo_queries(chain)
