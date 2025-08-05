import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.rag_pipeline import query_pipeline
import streamlit as st

st.title("🧠 RAG Chatbot")
query = st.text_input("Ask a question based on the documents")

if query:
    answer, sources = query_pipeline(query)
    st.write("### Answer")
    st.write(answer)

    st.write("### Sources")
    for doc in sources:
        st.write(f"- Source: {doc.metadata.get('source', 'unknown')}")
