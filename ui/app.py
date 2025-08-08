import sys
import os
import streamlit as st
import importlib.util

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import query module
spec = importlib.util.spec_from_file_location("query", "./embed/query.py")
query_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(query_module)

# Vector store from embed/query.py
vectorstore = query_module.vectorstore

from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

# Set Streamlit page config
st.set_page_config(page_title="RAG Chatbot", page_icon="⚽")
st.title("Hi! I'm Umpy")

# Initialize LLM
model_id = "google/flan-t5-base"  # You can upgrade to flan-t5-xl or mistral if needed
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt Template
from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a friendly, knowledgeable sports rules assistant chatbot.  
Your purpose is to answer questions about the official rules of football, basketball, cricket, tennis, and volleyball using only the information from the provided context.  
You can also engage naturally in casual conversation (e.g., greetings, small talk, thanks) in a polite and concise way.

### Context (from official sports rulebooks):
{context}

### User Question:
{question}

### Your Instructions:
1. **If the question is about sports rules:**
   - Give a clear, structured, and accurate answer.
   - Reference the relevant rule, law, or section from the context when applicable.
   - Keep explanations easy to understand for a general audience.
   - Provide examples or clarifications if they make the rule easier to grasp.
   - If the context contains partial information, answer with what is available and suggest what additional details might be needed.

2. **If the user greets you (e.g., "hi", "hello")**:
   - Respond warmly and invite them to ask a sports rules question.

3. **If the user thanks you (e.g., "thanks", "thank you")**:
   - Acknowledge politely and offer further assistance.

4. **If the context does not contain enough relevant information:**
   - Respond helpfully without fabricating facts.
   - Use a safe, natural fallback such as:
     > "I couldn’t find the exact rule in the material I have, but here’s what I can tell you based on related rules..."
     or
     > "The documents I have don’t specify that exact scenario, but here’s what usually applies according to the rules provided..."

5. **Maintain chatbot tone**:
   - Be concise but friendly.
   - Avoid overly formal legalistic tone unless the user specifically asks for it.
   - Adapt your style to the user's tone — more casual for casual chats, more formal for detailed rule queries.

### Begin your response below:
"""
)
# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Get user input
query = st.chat_input("Got a question about sports rules? Type it here...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Get response from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            try:
                result = qa_chain.invoke({"query": query})
                answer = result["result"]

                # Format answer
                st.markdown(f"**Answer:**\n\n{answer}")
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Source documents
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.markdown(f"**Document {i+1}:**")
                        st.markdown(doc.page_content)

            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
