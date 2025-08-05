# rag/rag_pipeline.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Load embedding
embedding = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")

# Load vector DB
db = FAISS.load_local("vector_store", embeddings=embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Load LLM
pipe = pipeline("text2text-generation", model="google/flan-t5-large", max_new_tokens=250)
llm = HuggingFacePipeline(pipeline=pipe)

# RAG QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

def query_pipeline(query):
    result = qa(query)
    return result["result"], result["source_documents"]