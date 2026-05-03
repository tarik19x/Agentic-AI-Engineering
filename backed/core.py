import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index",
    embedding=embeddings
)

# Initialize chat model
model = init_chat_model("gpt-5.2", model_provider="openai")


# 🔧 Retrieval Tool
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve relevant documentation to help answer user queries different News of Bangladesh.
    Returns both serialized text (for LLM) and raw documents (artifact).
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(query)

    # Serialize docs for LLM
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


# 🚀 Main RAG + Agent Pipeline
def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline using an agent with a retrieval tool.

    Args:
        query: User's question

    Returns:
        Dict containing:
            - answer (str)
            - context (List[Document])
    """

    system_prompt = (
    "You are a precise and reliable AI assistant that answers questions using retrieved context from a document database. "
    "You must base your answers ONLY on the provided context retrieved by the tool.\n\n"

    "Instructions:\n"
    "- Always use the retrieval tool before answering.\n"
    "- Only use information that appears in the retrieved context.\n"
    "- If the answer is not present in the context, say: "
    "'The provided context does not contain this information.'\n"
    "- Do NOT make up facts, numbers, or assumptions.\n"
    "- When answering numerical questions, ensure exact values from the context.\n"
    "- Keep answers concise but complete.\n"
    "- When possible, include the source (e.g., URL or document reference).\n\n"

    "Your goal is to provide accurate, grounded answers strictly based on retrieved documents."
)

    agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt=system_prompt
    )

    messages = [{"role": "user", "content": query}]

    response = agent.invoke({"messages": messages})

    # ✅ Extract final answer
    answer = response["messages"][-1].content

    # ✅ Extract retrieved documents from ToolMessage artifacts
    context_docs: List[Any] = []

    for message in response["messages"]:
        if isinstance(message, ToolMessage):
            artifact = getattr(message, "artifact", None)
            if isinstance(artifact, list):
                context_docs.extend(artifact)

    return {
        "answer": answer,
        "context": context_docs
    }


# ▶️ Entry point
if __name__ == "__main__":
    result = run_llm("What was the total amount of deposits in Islamic banking in 2025?")

    print("\n=== ANSWER ===\n")
    print(result["answer"])

    # print("\n=== CONTEXT DOCUMENTS ===\n")
    # for i, doc in enumerate(result["context"], 1):
    #     source = doc.metadata.get("source", "unknown")
    #     print(f"[{i}] Source: {source}")
    #     print(doc.page_content[:200], "...\n")  # preview