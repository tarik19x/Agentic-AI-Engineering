from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone, PineconeVectorStore

load_dotenv()



if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader(
    "C:/Users/tarik/Documents/GitHub/Agentic-AI-Engineering/mediumblog1.txt",
    encoding="utf-8"
)
    document = loader.load()

    print("splitting...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting to pinecone...")

  
    PineconeVectorStore.from_documents(texts,embeddings,index_name = os.environ['INDEX_NAME'])
    print("ingestion complete")