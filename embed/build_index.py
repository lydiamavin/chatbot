from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from ingest.chunk_documents import load_and_chunk
from utils.config import (
    PINECONE_API_KEY,
    PINECONE_ENV,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_CACHE,
)

load_dotenv()

documents = load_and_chunk()

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    cache_folder=EMBEDDING_MODEL_CACHE
)

pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(PINECONE_INDEX_NAME)

vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
    pinecone_api_key=PINECONE_API_KEY
)

vectorstore.add_documents(documents)

print(f"✅ Successfully embedded and added {len(documents)} documents to Pinecone.")

print("🔍 Testing similarity search:")
results = vectorstore.similarity_search("What is the policy?", k=2)
for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content[:200]}")