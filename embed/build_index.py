# embed/build_index.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ingest.chunk_documents import load_and_chunk

# Step 1: Load and chunk the document
docs = load_and_chunk("data/1f6114ff-431b-4820-883d-8ad5d4732df6_transcript.pdf")
print("this is the first chunk",docs[0].page_content[:500])

# Step 2: Load the embedding model
embedding = HuggingFaceEmbeddings(
    model_name="hkunlp/instructor-xl",
    cache_folder="./model_cache",
    model_kwargs={"revision": "main", "trust_remote_code": True}
)
print(f"Loaded {len(docs)} documents.")

# Step 3: Create the vector store
db = FAISS.from_documents(docs, embedding)

# Step 4: Save the vector store to a folder
db.save_local("vector_store")  # This should create index + faiss files
