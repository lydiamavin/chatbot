import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
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

# Load environment variables
load_dotenv()

# Step 1: Load and chunk PDF documents
documents = load_and_chunk()

# Step 2: Load the embedding model (Langchain 0.2+ compliant)
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    cache_folder=EMBEDDING_MODEL_CACHE
)

# Step 3: Initialize Pinecone (v3 SDK)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Step 4: Create index if not exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

# Step 5: Connect to the index (v3 uses `pc.Index` with capital "I")
index = pc.Index(PINECONE_INDEX_NAME)

# Step 6: Use Langchain’s Pinecone vectorstore wrapper
vectorstore = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")

# Step 7: Add documents
vectorstore.add_documents(documents)

print(f"✅ Successfully embedded and added {len(documents)} documents to Pinecone.")
