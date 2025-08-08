import os

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "rag-sports-index"

EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl"
EMBEDDING_MODEL_CACHE = "./model_cache"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PDF_DIRECTORY = "data"