The project is created. Here's what was built:

Project Structure

langchain-rag/
├── CLAUDE.md
├── requirements.txt
├── ingest.py          # PhasAISS
├── rag_chain.py       # Phase 2: load FAISS → build LCEL RAG chain
├── main.py            # Orch
├── .gitignore
└── documents/
    └── knowledge_base.txt   # Sample KB about RAG, LangChain, Bedrock, Claude

How to Run

# 1. Install dependencies
pip install -r requirements.t

# 2. Configure AWS credential-west-2)
aws configure

# 3. Ingest documents into the vector store
python ingest.py

# 4. Query
python main.py --query-only        # 5 demo questions
python main.py --interactive

Key Design Decisions

- FAISS as local vector DB — sts to vectorstore/ on disk
- Amazon Titan Embeddings V2 (amazon.titan-embed-text-v2:0) for generating document
vectors
- Claude Haiku (us.anthropic.claude-haiku-4-5-20251001-v1:0) as the LLM — same model
pattern as the sibling projec
- LCEL pipe chain — same composable | style used in firstchain/main.py
- RAG prompt instructs the mor isn't in the retrievedcontext