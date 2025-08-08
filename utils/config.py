import os

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = "rag-sports-index"

EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl"
EMBEDDING_MODEL_CACHE = "./model_cache"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PDF_DIRECTORY = "data"

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")