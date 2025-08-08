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
from sentence_transformers import SentenceTransformer, util
import numpy as np

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

spec = importlib.util.spec_from_file_location("query", "./embed/query.py")
query_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(query_module)
vectorstore = query_module.vectorstore

st.set_page_config(page_title="Sports Rules Chatbot", page_icon="⚽", layout="wide")
st.title("Hi! I'm Umpy – Your Sports Rules Assistant 🏆")

@st.cache_resource
def load_models():
    """Load and cache models for better performance"""
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=200,
        do_sample=True,
        temperature=0.1,
        top_p=0.85,
        repetition_penalty=1.15,
        early_stopping=True
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return llm, semantic_model

llm, semantic_model = load_models()

def is_greeting_or_thanks(query):
    """Check if query is just a greeting or thanks"""
    query_lower = query.lower().strip()
    greetings = ['hi', 'hello', 'hey']
    thanks = ['thanks', 'thank you', 'thx', 'ok', 'okay', 'good', 'great', 'nice']
    if query_lower in greetings or (len(query.split()) <= 2 and any(word in query_lower for word in greetings)):
        return "greeting"
    elif query_lower in thanks or (len(query.split()) <= 3 and any(word in query_lower for word in thanks)):
        return "thanks"
    return None

def extract_sport_from_query(query):
    sports_keywords = {
        'volleyball': ['volleyball', 'volley'],
        'basketball': ['basketball', 'basket'],
        'soccer': ['soccer', 'football'],
        'tennis': ['tennis'],
        'cricket': ['cricket'],
        'baseball': ['baseball'],
        'hockey': ['hockey'],
        'rugby': ['rugby']
    }
    
    query_lower = query.lower()
    for sport, keywords in sports_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return sport
    return None

def semantic_rerank_chunks(query, chunks, semantic_model, top_k=4):
    if not chunks:
        return chunks
    
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_embeddings = semantic_model.encode(chunk_texts, convert_to_tensor=True)
    
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]
    similarity_scores = similarities.cpu().numpy()
    sorted_indices = np.argsort(similarity_scores)[::-1]
    
    reranked_chunks = []
    for idx in sorted_indices[:top_k]:
        chunk = chunks[idx]
        chunk.metadata = chunk.metadata or {}
        chunk.metadata['similarity_score'] = float(similarity_scores[idx])
        reranked_chunks.append(chunk)
    
    return reranked_chunks

def advanced_chunk_processing(retrieved_docs, query):
    sport = extract_sport_from_query(query)
    
    query_words = len(query.split())
    if query_words > 10:
        chunk_size = 400
    elif query_words > 5:
        chunk_size = 350
    else:
        chunk_size = 300
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks = []
    for d in retrieved_docs:
        content = d.page_content
        if sport and sport not in content.lower():
            content = f"[{sport.title()} Rules] {content}"
        
        splits = splitter.split_text(content)
        for s in splits:
            if len(s.strip()) > 60:
                metadata = d.metadata.copy() if d.metadata else {}
                metadata['sport'] = sport
                all_chunks.append(Document(page_content=s, metadata=metadata))
    
    return all_chunks

def create_contextual_summary(chunk_text, query, llm, sport=None):
    sport_context = f" about {sport}" if sport else ""
    
    map_prompt = PromptTemplate(
        input_variables=["text", "question", "sport_context"],
        template=(
            "Extract the most relevant information from this sports rule passage that directly answers the question{sport_context}. "
            "Include specific numbers, measurements, conditions, or rules. "
            "Focus on factual details that help answer the question precisely. "
            "If the passage is not relevant to the question, respond with 'NOT_RELEVANT'.\n\n"
            "Rule Passage:\n{text}\n\n"
            "Question: {question}\n\n"
            "Key Information:"
        ),
    )
    
    try:
        chain = LLMChain(llm=llm, prompt=map_prompt)
        summary = chain.run({
            "text": chunk_text, 
            "question": query,
            "sport_context": sport_context
        }).strip()
        
        summary = re.sub(r'^(Key Information:|Answer:|Summary:)\s*', '', summary, flags=re.IGNORECASE)
        return summary.strip()
    except Exception:
        return chunk_text[:200] + "..."

def generate_enhanced_answer(context, query, llm, sport=None):
    sport_expertise = f"As a {sport} rules expert, " if sport else "As a sports rules expert, "
    
    enhanced_prompt = PromptTemplate(
        input_variables=["context", "question", "sport_expertise"],
        template=(
            "{sport_expertise}analyze the provided rule excerpts and answer the question with precision.\n\n"
            "Instructions:\n"
            "1. Read all the provided information carefully\n"
            "2. Identify the most relevant facts that answer the question\n"
            "3. Provide a clear, specific answer with exact numbers or details when available\n"
            "4. If there are multiple related rules, mention the most important ones\n"
            "5. Be concise but complete\n\n"
            "Rule Information:\n{context}\n\n"
            "Question: {question}\n\n"
            "Precise Answer:"
        ),
    )
    
    try:
        chain = LLMChain(llm=llm, prompt=enhanced_prompt)
        answer = chain.run({
            "context": context, 
            "question": query,
            "sport_expertise": sport_expertise
        }).strip()
        
        answer = re.sub(r'^(Precise Answer:|Answer:)\s*', '', answer, flags=re.IGNORECASE)
        return answer.strip()
    except Exception:
        return f"Based on the rules provided: {context[:300]}..."

def advanced_rag_pipeline(query, llm, vectorstore, semantic_model, k=8):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.get_relevant_documents(query)
    
    if not retrieved_docs:
        return {"answer": "I couldn't find relevant information in the sports rules database.", "sources": []}
    
    chunks = advanced_chunk_processing(retrieved_docs, query)
    reranked_chunks = semantic_rerank_chunks(query, chunks, semantic_model, top_k=6)
    sport = extract_sport_from_query(query)
    
    summaries = []
    for chunk in reranked_chunks:
        summary = create_contextual_summary(chunk.page_content, query, llm, sport)
        summaries.append({
            "summary": summary,
            "metadata": chunk.metadata,
            "orig_text": chunk.page_content,
            "similarity_score": chunk.metadata.get('similarity_score', 0)
        })
    
    relevant_summaries = []
    for i, s in enumerate(summaries):
        if ("NOT_RELEVANT" not in s["summary"].upper() and 
            len(s["summary"]) > 15 and 
            s["similarity_score"] > 0.1):
            source_ref = f"[Source {i+1}]"
            relevant_summaries.append(f"{source_ref} {s['summary']}")
    
    if not relevant_summaries:
        relevant_summaries = [f"[Source {i+1}] {s['orig_text'][:200]}..." 
                            for i, s in enumerate(summaries[:3])]
    
    combined_context = "\n\n".join(relevant_summaries[:6])
    final_answer = generate_enhanced_answer(combined_context, query, llm, sport)
    
    sources = []
    for i, s in enumerate(summaries[:6]):
        page = s["metadata"].get("page", "N/A") if s["metadata"] else "N/A"
        sport_detected = s["metadata"].get("sport", "General") if s["metadata"] else "General"
        similarity = s.get("similarity_score", 0)
        
        snippet = s["orig_text"].replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        
        sources.append({
            "index": i+1,
            "page": page,
            "sport": sport_detected,
            "similarity": f"{similarity:.2f}",
            "snippet": snippet,
            "summary": s["summary"]
        })
    
    return {
        "answer": final_answer,
        "sources": sources,
        "context_used": combined_context,
        "sport_detected": sport,
        "total_sources": len(retrieved_docs)
    }

k_value = 15 
left_col, center_col, right_col = st.columns([1, 2, 1])


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

bottom_placeholder = st.empty()
query = bottom_placeholder.chat_input("Ask me anything about sports rules...", key="chat_input")

if "process_question" in st.session_state:
    query = st.session_state.process_question
    del st.session_state.process_question

if query:
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
    
    query_type = is_greeting_or_thanks(query)
    if query_type == "greeting":
        reply = "Hello! 👋 How can I help you with sports rules today?"
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.stop()
    elif query_type == "thanks":
        reply = "You're welcome! 😊 Always here if you need more help."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                rag_result = advanced_rag_pipeline(query, llm, vectorstore, semantic_model, k=k_value)
                answer = rag_result['answer']
                st.markdown(f"\n\n{answer}")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                sport = rag_result.get('sport_detected', 'Not detected')
                st.info(f"🏀 Sport: **{sport.title() if sport else 'General'}**")
                
                with st.expander("📄 Detailed Sources", expanded=False):
                    for src in rag_result["sources"]:
                        if len(src['summary']) > 20 and "NOT_RELEVANT" not in src['summary'].upper():
                            st.markdown(f"**Key Info:** {src['summary']}")
                        st.markdown(f"**Source Text:** {src['snippet']}")
                        st.markdown(f"*Similarity Score: {src['similarity']}*")
                        st.markdown("---")
            except Exception as e:
                error_msg = "⚠️ Sorry, something went wrong while processing your request. Please try rephrasing your question."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})