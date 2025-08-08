import os
import traceback
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from pinecone import Pinecone
from transformers import AutoModelForSeq2SeqLM, pipeline, AutoTokenizer, AutoModelForCausalLM

from utils.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_CACHE,
)
PINECONE_INDEX_NAME="rag-sports-index"
load_dotenv()

# 1. Embedding model from langchain_huggingface (no torch/sentence_transformers)
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    cache_folder=EMBEDDING_MODEL_CACHE,
    model_kwargs={"device": "cpu"}
)

# 2. Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# 3. Vector store
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_model,
    text_key="text"
)

# 4. Local model for text generation using transformers pipeline
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

llm = HuggingFacePipeline(pipeline=pipe)

# 5. RAG QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            result = qa_chain.invoke({"query": query})
            print(result["result"])
        except Exception as e:
            print("\n❌ Error:", str(e))
            traceback.print_exc()
