# app.py
import sys
import os
import streamlit as st
import importlib.util
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_core.documents import Document
import re

# Load environment
load_dotenv()

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import vectorstore from embed/query.py
spec = importlib.util.spec_from_file_location("query", "./embed/query.py")
query_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(query_module)
vectorstore = query_module.vectorstore

# Streamlit page config
st.set_page_config(page_title="RAG Sports Rules Chatbot", page_icon="⚽")
st.title("Hi! I'm Umpy – Your Sports Rules Assistant")

# Load LLM with better parameters
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Improved pipeline with better generation parameters
pipe = pipeline(
    "text2text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=256,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
    repetition_penalty=1.1
)
llm = HuggingFacePipeline(pipeline=pipe)

def clean_and_validate_summary(summary_text, original_text):
    """Clean and validate summary text"""
    summary = summary_text.strip()
    
    # Remove common artifacts and clean up
    summary = re.sub(r'^(Summary:|Answer:)\s*', '', summary, flags=re.IGNORECASE)
    summary = summary.strip()
    
    # If summary is too short or seems invalid, use a portion of original text
    if len(summary) < 10 or summary.lower() in ['no', 'none', '']:
        return original_text[:200] + "..."
    
    return summary

def answer_with_improved_summaries(query, llm, vectorstore, k=6):
    """Improved RAG function with better summarization and answer generation"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.get_relevant_documents(query)

    # Use smaller chunks for better granularity
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    small_chunks = []
    for d in retrieved_docs:
        splits = splitter.split_text(d.page_content)
        for s in splits:
            if len(s.strip()) > 50:  # Filter out very small chunks
                small_chunks.append(Document(page_content=s, metadata=d.metadata or {}))

    # Improved summarization prompt
    map_prompt = PromptTemplate(
        input_variables=["text", "question"],
        template=(
            "Read this sports rules passage and extract only the specific information that answers the question. "
            "Be precise and include numbers, rules, or facts that directly relate to the question. "
            "If the passage doesn't contain relevant information, write 'NOT_RELEVANT'.\n\n"
            "Passage: {text}\n\n"
            "Question: {question}\n\n"
            "Relevant information:"
        ),
    )
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    summaries = []
    for chunk in small_chunks:
        try:
            summary_raw = map_chain.run({"text": chunk.page_content, "question": query})
            summary = clean_and_validate_summary(summary_raw, chunk.page_content)
        except Exception as e:
            st.warning(f"Error in summarization: {e}")
            summary = chunk.page_content[:200] + "..."
        
        summaries.append({
            "summary": summary, 
            "metadata": chunk.metadata, 
            "orig_text": chunk.page_content
        })

    # Filter and rank summaries
    useful_summaries = []
    for i, x in enumerate(summaries):
        if "NOT_RELEVANT" not in x["summary"].upper() and len(x["summary"].strip()) > 20:
            useful_summaries.append(f"Source [{i+1}]: {x['summary']}")

    # Limit context length but ensure we have good information
    combined_context = "\n\n".join(useful_summaries[:8])
    if len(combined_context) > 2500:
        combined_context = combined_context[:2500] + "..."

    # Improved final answer prompt with examples
    final_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a sports rules expert. Use ONLY the provided context to answer the question accurately. "
            "Give a clear, direct answer with specific numbers or rules when available. "
            "Reference sources using [Source X] when mentioning specific information.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Direct Answer:"
        ),
    )
    
    final_chain = LLMChain(llm=llm, prompt=final_prompt)
    
    try:
        answer_raw = final_chain.run({"context": combined_context, "question": query})
        # Clean up the answer
        answer = re.sub(r'^(Answer:|Direct Answer:)\s*', '', answer_raw.strip(), flags=re.IGNORECASE)
        answer = answer.strip()
        
        # Fallback if answer is too short or seems wrong
        if len(answer) < 10:
            answer = "Based on the retrieved information: " + combined_context[:200] + "..."
            
    except Exception as e:
        st.error(f"Error generating final answer: {e}")
        answer = "I found relevant information but encountered an error generating the final answer."

    # Prepare source information
    sources = []
    for i, s in enumerate(summaries[:6]):
        page = s["metadata"].get("page", "N/A") if s["metadata"] else "N/A"
        snippet = s["orig_text"].replace("\n", " ").strip()
        if len(snippet) > 250:
            snippet = snippet[:250] + "..."
        
        sources.append({
            "index": i+1, 
            "page": page, 
            "snippet": snippet, 
            "summary": s["summary"]
        })

    return {
        "answer": answer, 
        "sources": sources,
        "context_used": combined_context  # For debugging
    }

# Enhanced debug mode
DEBUG_MODE = st.sidebar.checkbox("Show Debug Info", value=False)

# ---- Streamlit Chat ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Got a question about sports rules? Type it here...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing sports rules..."):
            try:
                rag_out = answer_with_improved_summaries(query, llm, vectorstore, k=6)
                
                # Display the answer
                st.markdown(f"\n\n{rag_out['answer']}")
                st.session_state.messages.append({"role": "assistant", "content": rag_out["answer"]})

                # Source documents with better formatting
                with st.expander("📄 Source Documents", expanded=False):
                    for src in rag_out["sources"]:
                        st.markdown(f"**[{src['index']}] Page {src['page']}**")
                        
                        # Only show summary if it's meaningful
                        if len(src['summary']) > 20 and "NOT_RELEVANT" not in src['summary'].upper():
                            st.markdown(f"*Key Info:* {src['summary']}")
                        
                        st.markdown(f"*Source Text:* {src['snippet']}")
                        st.markdown("---")
                
                # Debug information
                if DEBUG_MODE:
                    with st.expander("🔍 Debug Information"):
                        st.markdown("**Context sent to LLM:**")
                        st.text(rag_out.get('context_used', 'N/A'))
                        
            except Exception as e:
                error_msg = f"⚠️ I encountered an error while processing your question: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                if DEBUG_MODE:
                    st.exception(e)

with st.sidebar:
    
    st.markdown("### 🏆 Example Questions")
    example_questions = [
        "How many players in a volleyball team?",
        "What is the offside rule in soccer?",
        "How many fouls before ejection in basketball?",
        "What is the size of a tennis court?"
    ]
    
    for eq in example_questions:
        if st.button(eq, key=eq):
            st.session_state.messages.append({"role": "user", "content": eq})
            st.rerun()