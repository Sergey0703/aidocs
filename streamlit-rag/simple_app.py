#!/usr/bin/env python3
import streamlit as st
import time
import psycopg2
from dotenv import load_dotenv
import os
from llama_index.llms.ollama import Ollama

load_dotenv()

st.title("Simple RAG - Fixed Version")

# Simple SQL search + Direct LLM
def simple_search(query):
    conn = psycopg2.connect(os.getenv("SUPABASE_CONNECTION_STRING"))
    cur = conn.cursor()
    
    # Search for Breeda Daly specifically
    if 'breeda' in query.lower() or 'daly' in query.lower():
        cur.execute("""
            SELECT metadata->>'text' 
            FROM vecs.documents 
            WHERE LOWER(metadata->>'text') LIKE %s
               OR LOWER(metadata->>'text') LIKE %s
            LIMIT 3
        """, ('%breeda%', '%daly%'))
    else:
        # General search
        cur.execute("""
            SELECT metadata->>'text' 
            FROM vecs.documents 
            WHERE LOWER(metadata->>'text') LIKE %s
            LIMIT 3
        """, (f'%{query.lower()}%',))
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return [row[0] for row in results]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask about documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        start_time = time.time()
        
        # 1. Search documents
        search_start = time.time()
        documents = simple_search(prompt)
        search_time = time.time() - search_start
        
        # 2. Create context
        if documents:
            # OPTIMIZATION: Limit context size to prevent timeouts
            context = "\n\n".join(documents[:2])[:2000]  # Limit to 2000 chars
            context_preview = f"Found {len(documents)} documents with relevant information."
        else:
            context = "No relevant documents found in the database."
            context_preview = "No documents found."
        
        # 3. Single LLM call with timeout handling
        llm_start = time.time()
        try:
            # OPTIMIZATION: Configure LLM with shorter timeouts and limits
            llm = Ollama(
                model="llama3.2:3b", 
                base_url="http://localhost:11434",
                request_timeout=60,  # 60 second timeout
                additional_kwargs={
                    "temperature": 0.1,
                    "num_predict": 100,  # Limit response length
                    "top_k": 10,
                    "top_p": 0.9,
                    "stop": ["</answer>", "\n\n\n"]  # Early stopping
                }
            )
            
            if documents:
                # OPTIMIZATION: Shorter, more focused prompt
                full_prompt = f"""Context: {context}

Question: {prompt}

Provide a brief answer based on the context above (max 2-3 sentences):"""
            else:
                full_prompt = f"""No documents found for: {prompt}

Provide a brief explanation that the information is not available."""
            
            response = llm.complete(full_prompt)
            response_text = response.text.strip()
            
            # Fallback if response is too long
            if len(response_text) > 500:
                response_text = response_text[:500] + "..."
            
        except Exception as e:
            error_msg = str(e)
            if "timed out" in error_msg.lower():
                response_text = "?? Response timed out. The system found relevant documents but LLM processing took too long. Try a shorter question."
            else:
                response_text = f"?? Error: {error_msg}"
        
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time
        
        # Display response
        st.write(response_text)
        
        # Display timing info with colors
        if total_time < 30:
            timing_color = "??"
        elif total_time < 60:
            timing_color = "??"
        else:
            timing_color = "??"
        
        st.write(f"**{timing_color} Timing:** Total {total_time:.1f}s | Search {search_time:.1f}s | LLM {llm_time:.1f}s")
        st.write(f"**?? Documents found:** {len(documents)}")
        st.write(f"**?? Context preview:** {context_preview}")
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"{response_text}\n\n**Stats:** {total_time:.1f}s total, {len(documents)} docs found"
        })