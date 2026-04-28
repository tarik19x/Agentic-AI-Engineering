import os

from dotenv import load_dotenv
import langchain_core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

vectorstore = PineconeVectorStore(
    index_name=os.environ['INDEX_PINECONE_2'],
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})




prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])





def create_retrieval_chain_with_lcel():
    
    retrival_chain = (
        RunnablePassthrough.assign(context=itemgetter("question")|retriever|format_docs)
     |prompt_template | llm | StrOutputParser() )
    
    return retrival_chain
    

if __name__ == "__main__":
    print("Testing Context Retrieval...")
    query = "What do you know about the SCORE framework?"
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)