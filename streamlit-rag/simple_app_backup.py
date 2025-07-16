#!/usr/bin/env python3
import streamlit as st
import time
import psycopg2
from dotenv import load_dotenv
import os
from llama_index.llms.ollama import Ollama

load_dotenv()

st.title("Simple RAG - NO VECTORS")

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
            context = "\n\n".join(documents[:2])
            context_preview = f"Found {len(documents)} documents with relevant information."
        else:
            context = "No relevant documents found in the database."
            context_preview = "No documents found."
        
        # 3. Single LLM call
        llm_start = time.time()
        try:
            llm = Ollama(model="llama3.2:3b", base_url="http://localhost:11434")
            
            if documents:
                full_prompt = f"""Based on this context from documents:

{context}

Question: {prompt}

Please provide a clear answer based on the information above. If the information is not sufficient, say so."""
            else:
                full_prompt = f"""No relevant documents were found for: {prompt}

Please provide a general response explaining that the information is not available in the current document database."""
            
            response = llm.complete(full_prompt)
            response_text = response.text
            
        except Exception as e:
            response_text = f"Error generating response: {str(e)}"
        
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time
        
        # Display response
        st.write(response_text)
        
        # Display timing info
        st.write(f"**Timing:** Total {total_time:.1f}s | Search {search_time:.1f}s | LLM {llm_time:.1f}s")
        st.write(f"**Documents found:** {len(documents)}")
        st.write(f"**Context preview:** {context_preview}")
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"{response_text}\n\n**Stats:** {total_time:.1f}s total, {len(documents)} docs found"
        })
