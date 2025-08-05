import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.rag_pipeline import query_pipeline  # Your custom query function

st.set_page_config(page_title="RAG Chatbot", page_icon="🧠")

st.title("Hey there!")

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ask a question based on the documents..."):
    # Display user's message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call your RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = query_pipeline(query)
            st.markdown(f"**Answer:** {answer}")

            # Optionally show sources
            if sources:
                st.markdown("**Sources:**")
                for doc in sources:
                    source = doc.metadata.get("source", "unknown")
                    st.markdown(f"- {source}")

    # Save assistant's response
    st.session_state.messages.append({"role": "assistant", "content": answer})
