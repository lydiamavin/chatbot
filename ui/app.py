
import sys
import os
import streamlit as st

# Force add current directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Now import after setting path
import importlib.util

spec = importlib.util.spec_from_file_location("query", "./embed/query.py")
query_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(query_module)

vectorstore = query_module.vectorstore
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline

# Setup page
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG-based Chatbot")

# Initialize LLM
model_id = "google/flan-t5-base"  # You can change this if needed

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipe)

# Build RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Input box
query = st.text_input("Ask a question:", placeholder="e.g. What is PMAY?")
if query:
    with st.spinner("Generating answer..."):
        try:
            result = qa_chain.invoke({"query": query})
            st.markdown("### ✅ Answer:")
            st.write(result["result"])

            # Optional: Show source documents
            with st.expander("📄 Source Documents"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Document {i+1}:**")
                    st.write(doc.page_content)
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
