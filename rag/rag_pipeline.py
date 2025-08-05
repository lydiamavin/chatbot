from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()

# Load the same embedding model used during indexing
embedding = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")

# Load the vector store
db = FAISS.load_local("vector_store", embeddings=embedding, allow_dangerous_deserialization=True)

# Set up retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Set up LLM
pipe = pipeline("text2text-generation", model="google/flan-t5-large")
llm = HuggingFacePipeline(pipeline=pipe)

# Set up QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Ask questions interactively
while True:
    query = input("Ask a question: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = qa(query)
    print("\nAnswer:")
    print(response["result"])
    print("\nSources:")
    for doc in response["source_documents"]:
        print(doc.metadata)