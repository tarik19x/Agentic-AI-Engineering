from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

# load env variables
load_dotenv()

# init pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = os.getenv("INDEX_PINECONE_2")

existing_indexes = [i.name for i in pc.list_indexes()]

# delete old index if exists (important)
if index_name in existing_indexes:
    pc.delete_index(index_name)

pc.create_index(
    name=index_name,
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"   # ✅ FIXED
    )
)

# load pdf
loader = PyPDFLoader("SCORE.pdf")
documents = loader.load()

# split text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# embeddings
embeddings = OpenAIEmbeddings()

# upload to pinecone
vectorstore = PineconeVectorStore.from_documents(
    docs,
    embedding=embeddings,
    index_name=index_name
)

# connect later for retrieval
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

retriever = vectorstore.as_retriever()

results = retriever.invoke("What is this document about?")

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)