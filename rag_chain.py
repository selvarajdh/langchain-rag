"""
RAG chain: loads the FAISS vector store and builds an LCEL retrieval-augmented chain.
Falls back to the LLM's general knowledge when the query doesn't match the knowledge base.
"""

from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
AWS_REGION = "us-west-2"
VECTOR_STORE_PATH = "vectorstore"
TOP_K_RESULTS = 3
# L2 distance threshold: above this score the knowledge base is considered irrelevant
# (0 = identical, 2 = opposite for normalised embeddings; 1.0 is a reasonable cutoff)
SIMILARITY_THRESHOLD = 1.0

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the question using only the provided context.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])

FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. The question was not found in the knowledge base. "
        "Answer using your general knowledge.",
    ),
    ("human", "{question}"),
])


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(store_path: str = VECTOR_STORE_PATH):
    """
    Returns an LCEL chain: question (str) → answer (str).
    Uses RAG when the knowledge base contains relevant context; falls back to the
    LLM's general knowledge when the best match exceeds SIMILARITY_THRESHOLD.
    """
    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        region_name=AWS_REGION,
    )
    vector_store = FAISS.load_local(
        store_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    llm = ChatBedrockConverse(
        model=LLM_MODEL_ID,
        region_name=AWS_REGION,
    )
    output_parser = StrOutputParser()

    def _invoke(question: str) -> str:
        docs_with_scores = vector_store.similarity_search_with_score(question, k=TOP_K_RESULTS)
        best_score = docs_with_scores[0][1] if docs_with_scores else float("inf")

        if best_score <= SIMILARITY_THRESHOLD:
            docs = [doc for doc, _ in docs_with_scores]
            return (RAG_PROMPT | llm | output_parser).invoke({
                "context": _format_docs(docs),
                "question": question,
            })

        # No relevant context found — use LLM general knowledge
        print(f"[Fallback] No relevant match in knowledge base (best score: {best_score:.3f}). "
              "Using LLM general knowledge.")
        return (FALLBACK_PROMPT | llm | output_parser).invoke({"question": question})

    return RunnableLambda(_invoke)


if __name__ == "__main__":
    chain = build_rag_chain()
    questions = [
        "What is Retrieval-Augmented Generation?",
        "What is the capital of France?",  # not in knowledge base → fallback
    ]
    for question in questions:
        print(f"Q: {question}")
        print(f"A: {chain.invoke(question)}")
        print("-" * 60)