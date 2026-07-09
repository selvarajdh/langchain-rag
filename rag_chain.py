"""
RAG chain: loads the FAISS vector store and builds an LCEL retrieval-augmented chain.
"""

from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
AWS_REGION = "us-west-2"
VECTOR_STORE_PATH = "vectorstore"
TOP_K_RESULTS = 3

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If the answer is not found in the context, say: "
        "'I don't have enough information in the knowledge base to answer that question.'\n\n"
        "Context:\n{context}",
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
    Requires the vector store to exist at `store_path` (run ingest.py first).
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
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    llm = ChatBedrockConverse(
        model=LLM_MODEL_ID,
        region_name=AWS_REGION,
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    chain = build_rag_chain()
    question = "What is Retrieval-Augmented Generation?"
    print(f"Q: {question}")
    print(f"A: {chain.invoke(question)}")
