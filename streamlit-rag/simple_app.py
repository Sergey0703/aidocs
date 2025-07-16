#!/usr/bin/env python3
import streamlit as st
import time
import psycopg2
from dotenv import load_dotenv
import os
from llama_index.llms.ollama import Ollama

load_dotenv()

st.title("RAG System - Precision Search")

# Improved search with better precision and ranking
def precision_search(query):
    conn = psycopg2.connect(os.getenv("SUPABASE_CONNECTION_STRING"))
    cur = conn.cursor()
    
    # Clean and prepare query
    words = [word.strip().lower() for word in query.split() if len(word.strip()) > 1]
    
    if not words:
        cur.close()
        conn.close()
        return []
    
    if len(words) == 1:
        # Single word search with word boundaries
        cur.execute("""
            SELECT metadata->>'text', metadata->>'file_path', metadata->>'file_name'
            FROM vecs.documents 
            WHERE LOWER(metadata->>'text') ~ %s
            ORDER BY LENGTH(metadata->>'text') ASC
            LIMIT 5
        """, (f'\\\\b{words[0]}\\\\b',))
    else:
        # Multi-word search with phrase priority
        full_phrase = ' '.join(words)
        
        # Build search conditions
        search_conditions = []
        search_params = []
        
        # Priority 1: Exact phrase match
        search_conditions.append("LOWER(metadata->>'text') LIKE %s")
        search_params.append(f'%{full_phrase}%')
        
        # Priority 2: All words present (AND logic)
        and_conditions = []
        for word in words:
            if len(word) > 2:
                and_conditions.append("LOWER(metadata->>'text') ~ %s")
                search_params.append(f'\\\\b{word}\\\\b')
        
        if and_conditions:
            search_conditions.append(f"({' AND '.join(and_conditions)})")
        
        # Priority 3: Any word present (OR logic)
        or_conditions = []
        for word in words:
            if len(word) > 2:
                or_conditions.append("LOWER(metadata->>'text') ~ %s")
                search_params.append(f'\\\\b{word}\\\\b')
        
        if or_conditions:
            search_conditions.append(f"({' OR '.join(or_conditions)})")
        
        # Final query with ranking
        query_sql = f"""
            SELECT 
                metadata->>'text', 
                metadata->>'file_path', 
                metadata->>'file_name',
                CASE 
                    WHEN LOWER(metadata->>'text') LIKE %s THEN 1
                    WHEN {' AND '.join([f"LOWER(metadata->>'text') ~ %s" for word in words if len(word) > 2])} THEN 2
                    ELSE 3
                END as relevance_score
            FROM vecs.documents 
            WHERE {' OR '.join(search_conditions)}
            ORDER BY relevance_score ASC, LENGTH(metadata->>'text') ASC
            LIMIT 5
        """
        
        # Add phrase match parameter for ranking
        final_params = [f'%{full_phrase}%']
        # Add AND ranking parameters
        for word in words:
            if len(word) > 2:
                final_params.append(f'\\\\b{word}\\\\b')
        # Add original search parameters
        final_params.extend(search_params)
        
        cur.execute(query_sql, final_params)
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    # Return only text, file_path, file_name (remove relevance score if present)
    return [(row[0], row[1], row[2]) for row in results]

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
        
        # 1. Search documents with precision
        search_start = time.time()
        documents = precision_search(prompt)
        search_time = time.time() - search_start
        
        # Display document roster
        if documents:
            with st.expander("?? Found Documents", expanded=True):
                st.write(f"**Found {len(documents)} relevant document chunks (ranked by relevance):**")
                for i, (text, file_path, file_name) in enumerate(documents, 1):
                    file_display = file_path or file_name or "Unknown file"
                    
                    # Better preview - show more relevant parts
                    preview_text = text.replace('\n', ' ').strip()
                    if len(preview_text) > 200:
                        # Try to find the query terms in the text for better preview
                        query_words = [word.lower() for word in prompt.split() if len(word) > 2]
                        best_start = 0
                        for word in query_words:
                            word_pos = preview_text.lower().find(word)
                            if word_pos != -1:
                                best_start = max(0, word_pos - 50)
                                break
                        
                        preview = preview_text[best_start:best_start+200] + "..."
                        if best_start > 0:
                            preview = "..." + preview
                    else:
                        preview = preview_text
                    
                    st.write(f"**{i}.** `{file_display}`")
                    st.write(f"   *Preview:* {preview}")
                    st.write("---")
        
        # 2. Create enhanced context
        if documents:
            # Use more context and prioritize first 2 results
            context_parts = []
            for i, (text, file_path, file_name) in enumerate(documents[:3]):
                file_display = file_path or file_name or f"Document {i+1}"
                context_parts.append(f"[Source: {file_display}]\n{text}")
            
            context = "\n\n".join(context_parts)[:3000]  # Increased context size
            context_preview = f"Found {len(documents)} documents with relevant information."
        else:
            context = "No relevant documents found in the database."
            context_preview = "No documents found."
        
        # 3. Enhanced LLM call with better prompting
        llm_start = time.time()
        try:
            llm = Ollama(
                model="llama3.2:3b", 
                base_url="http://localhost:11434",
                request_timeout=90,  # Increased timeout
                additional_kwargs={
                    "temperature": 0.2,
                    "num_predict": 200,  # More detailed responses
                    "top_k": 10,
                    "top_p": 0.9,
                    "stop": ["</answer>", "\n\n---", "Source:"]
                }
            )
            
            if documents:
                # Enhanced prompt with better instructions
                full_prompt = f"""You are a helpful assistant analyzing documents. Based on the following context from multiple documents, provide a comprehensive answer to the user's question.

Context from documents:
{context}

User Question: {prompt}

Instructions:
1. Provide a clear, informative answer based on the document content
2. Include specific details like names, dates, locations, certificate numbers if mentioned
3. If you find information about the person/topic, present it in a structured way
4. If information is incomplete, mention what is available
5. Be specific about what you found in the documents

Answer:"""
            else:
                full_prompt = f"""No documents found containing information about: {prompt}

Please provide a brief explanation that the requested information is not available in the current document database."""
            
            response = llm.complete(full_prompt)
            response_text = response.text.strip()
            
            # Clean up response
            if len(response_text) > 1000:
                response_text = response_text[:1000] + "..."
            
        except Exception as e:
            error_msg = str(e)
            if "timed out" in error_msg.lower():
                response_text = "?? Response timed out. The system found relevant documents but processing took too long. Please try a more specific question."
            else:
                response_text = f"?? Error: {error_msg}"
        
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time
        
        # Display response
        st.write(response_text)
        
        # Enhanced timing info
        if total_time < 30:
            timing_color = "??"
        elif total_time < 60:
            timing_color = "??"
        else:
            timing_color = "??"
        
        st.write(f"**{timing_color} Performance:** Total {total_time:.1f}s | Search {search_time:.1f}s | LLM {llm_time:.1f}s")
        st.write(f"**?? Documents found:** {len(documents)}")
        st.write(f"**?? Context status:** {context_preview}")
        
        # Add search quality indicator
        if documents:
            relevance_indicators = []
            query_words = set(prompt.lower().split())
            
            for text, _, _ in documents[:3]:
                text_words = set(text.lower().split())
                overlap = len(query_words.intersection(text_words))
                relevance_indicators.append(overlap)
            
            avg_relevance = sum(relevance_indicators) / len(relevance_indicators) if relevance_indicators else 0
            if avg_relevance > 2:
                quality_indicator = "?? High relevance"
            elif avg_relevance > 1:
                quality_indicator = "?? Medium relevance"
            else:
                quality_indicator = "?? Low relevance - try different keywords"
            
            st.write(f"**?? Search quality:** {quality_indicator}")
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"{response_text}\n\n**Stats:** {total_time:.1f}s total, {len(documents)} docs found"
        })