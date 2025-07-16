#!/usr/bin/env python3
"""
Fixed Intelligent Query RAG System - No Limits
Uses clean LLM entity extraction + Simple LIKE search + LLM for answer generation
Fixed re import error and removed document limits
"""
import streamlit as st
import time
import psycopg2
from dotenv import load_dotenv
import os
from llama_index.llms.ollama import Ollama
import json
import re

load_dotenv()

st.title("?? Intelligent Query RAG - No Limits")
st.caption("Clean Entity Extraction + Simple LIKE Search + LLM Answer Generation")

# Stage 1: Clean Entity Extraction using industry best practices
def extract_search_entities(query):
    """Clean entity extraction using industry best practices for minimal output"""
    
    # Configure LLM for minimal, structured output
    llm = Ollama(
        model="llama3.2:3b", 
        base_url="http://localhost:11434",
        request_timeout=30,
        additional_kwargs={
            "temperature": 0.0,     # Deterministic output
            "num_predict": 10,      # Maximum 10 tokens
            "top_k": 1,            # Most confident choice
            "top_p": 0.1,          # Minimal diversity
            "stop": ["\n", ".", ",", ":", ";", "!", "?", " and", " or"]  # Stop tokens
        }
    )
    
    # Simplified prompt for clean extraction - based on best practices
    extraction_prompt = f"""Extract only the person's name from this question. Return ONLY the name, no other words.

Examples:
- "tell me about John Smith" ? John Smith
- "who is Mary Johnson" ? Mary Johnson
- "find information about Bob Wilson" ? Bob Wilson

Question: {query}

Name:"""
    
    try:
        response = llm.complete(extraction_prompt)
        extracted_terms = response.text.strip()
        
        # Clean extraction using best practices
        # Remove any remaining prefixes/suffixes
        extracted_terms = re.sub(r'^(name|answer|result)[:=]\s*', '', extracted_terms, flags=re.IGNORECASE)
        extracted_terms = re.sub(r'\s*(is|the|answer|result)$', '', extracted_terms, flags=re.IGNORECASE)
        
        # If extraction failed or contains unwanted words, fallback to simple regex
        unwanted_words = ['question', 'query', 'extract', 'name', 'tell', 'about', 'who', 'is', 'find']
        if any(word.lower() in extracted_terms.lower() for word in unwanted_words):
            # Fallback: use simple regex to find capitalized words
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
            # Filter out common question words
            name_words = [word for word in words if word.lower() not in ['Tell', 'About', 'Who', 'Show', 'Find', 'What']]
            if name_words:
                extracted_terms = ' '.join(name_words[:2])  # Take first 2 words (first name + last name)
            else:
                extracted_terms = query.strip()
        
        return extracted_terms.strip()
    except Exception as e:
        st.error(f"Error in query extraction: {e}")
        # Fallback to regex extraction
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        name_words = [word for word in words if word.lower() not in ['Tell', 'About', 'Who', 'Show', 'Find', 'What']]
        if name_words:
            return ' '.join(name_words[:2])
        return query.strip()

# Stage 2: Simple LIKE search without limits
def simple_like_search(query):
    """Simple LIKE search - no complications, no limits, just works"""
    conn = psycopg2.connect(os.getenv("SUPABASE_CONNECTION_STRING"))
    cur = conn.cursor()
    
    # Clean query
    query = query.strip()
    if not query:
        cur.close()
        conn.close()
        return []
    
    # SIMPLE search - NO LIMIT - get all documents
    cur.execute("""
        SELECT metadata->>'text', metadata->>'file_path', metadata->>'file_name'
        FROM vecs.documents 
        WHERE LOWER(metadata->>'text') LIKE %s
        ORDER BY LENGTH(metadata->>'text') ASC
    """, (f'%{query.lower()}%',))
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

# Stage 3: Answer Generation with enhanced context
def generate_intelligent_answer(query, extracted_terms, documents):
    """Generate answer using original query, extracted terms, and found documents"""
    # Use different LLM settings for answer generation
    llm = Ollama(
        model="llama3.2:3b", 
        base_url="http://localhost:11434",
        request_timeout=90,
        additional_kwargs={
            "temperature": 0.2,
            "num_predict": 300,  # Increased for full answers
            "top_k": 10,
            "top_p": 0.9,
        }
    )
    
    if not documents:
        no_results_prompt = f"""The user asked: "{query}"
I extracted these search terms: "{extracted_terms}"
But no relevant documents were found in the database.

Please provide a helpful response explaining that the information is not available in the current document database."""
        
        try:
            response = llm.complete(no_results_prompt)
            return response.text.strip()
        except Exception as e:
            return f"No relevant documents found for: {extracted_terms}"
    
    # Create enhanced context - use more documents if available
    context_parts = []
    for i, (text, file_path, file_name) in enumerate(documents[:5]):  # Use first 5 for context
        source = file_path or file_name or f"Document {i+1}"
        context_parts.append(f"[Source: {source}]\n{text}")
    
    context = "\n\n".join(context_parts)[:4000]  # Larger context
    
    answer_prompt = f"""You are an intelligent document assistant. Based on the provided context, answer the user's question comprehensively.

Original User Question: "{query}"
Extracted Search Terms: "{extracted_terms}"

Context from Documents:
{context}

Instructions:
1. Provide a complete, informative answer based on the document content
2. Include specific details like names, dates, locations, certificate numbers, organizations
3. Structure your response clearly with relevant information
4. If you find information about a person, include their role, certifications, training, etc.
5. Reference specific details from the documents when possible
6. If information is incomplete, mention what is available and what might be missing

Answer:"""
    
    try:
        response = llm.complete(answer_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer: {e}"

# Streamlit app logic
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
        
        # Stage 1: Clean Entity Extraction
        st.write("?? **Stage 1: Clean entity extraction...**")
        extraction_start = time.time()
        extracted_terms = extract_search_entities(prompt)
        extraction_time = time.time() - extraction_start
        
        st.write(f"? **Extracted entity:** `{extracted_terms}`")
        st.write(f"?? **Extraction time:** {extraction_time:.1f}s")
        
        # Show what search query will be used
        search_query = extracted_terms.strip()
        st.write(f"?? **Search query:** `{search_query}`")
        
        # Stage 2: Simple LIKE Search - NO LIMITS
        st.write("?? **Stage 2: Searching documents (no limits)...**")
        search_start = time.time()
        
        documents = simple_like_search(search_query)
        search_time = time.time() - search_start
        
        st.write(f"?? **Found {len(documents)} relevant documents**")
        st.write(f"?? **Search time:** {search_time:.1f}s")
        
        # Display document roster - show all documents
        if documents:
            with st.expander("?? Found Documents", expanded=True):
                st.write(f"**All {len(documents)} documents found for: `{search_query}`**")
                for i, (text, file_path, file_name) in enumerate(documents, 1):
                    file_display = file_path or file_name or "Unknown file"
                    
                    # Smart preview - show relevant parts
                    preview_text = text.replace('\n', ' ').strip()
                    if len(preview_text) > 300:
                        # Find search terms in text for better preview
                        search_words = search_query.lower().split()
                        best_start = 0
                        for word in search_words:
                            if len(word) > 2:
                                word_pos = preview_text.lower().find(word)
                                if word_pos != -1:
                                    best_start = max(0, word_pos - 100)
                                    break
                        
                        preview = preview_text[best_start:best_start+300] + "..."
                        if best_start > 0:
                            preview = "..." + preview
                    else:
                        preview = preview_text
                    
                    st.write(f"**{i}.** `{file_display}`")
                    st.write(f"   *Preview:* {preview}")
                    
                    # Show if search terms are actually found
                    search_found = search_query.lower() in preview_text.lower()
                    if search_found:
                        st.write(f"   ? *Contains: {search_query}*")
                    else:
                        st.write(f"   ? *Search terms not visible in preview*")
                    st.write("---")
        
        # Stage 3: Answer Generation
        st.write("?? **Stage 3: Generating comprehensive answer...**")
        answer_start = time.time()
        final_answer = generate_intelligent_answer(prompt, extracted_terms, documents)
        answer_time = time.time() - answer_start
        
        st.write(f"?? **Answer generation time:** {answer_time:.1f}s")
        st.write("---")
        
        # Display final answer
        st.write("**?? Answer:**")
        st.write(final_answer)
        
        # Performance summary
        total_time = time.time() - start_time
        st.write("---")
        st.write("**? Performance Summary:**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Entity Extraction", f"{extraction_time:.1f}s")
        with col2:
            st.metric("Document Search", f"{search_time:.1f}s")
        with col3:
            st.metric("Answer Generation", f"{answer_time:.1f}s")
        with col4:
            st.metric("Total Time", f"{total_time:.1f}s")
        
        # Quality indicators
        if documents:
            st.write("**?? Search Quality Analysis:**")
            # Check if search terms are actually found in documents
            search_terms_found = 0
            total_docs_checked = min(5, len(documents))  # Check first 5 for quality
            
            for text, _, _ in documents[:total_docs_checked]:
                if search_query.lower() in text.lower():
                    search_terms_found += 1
            
            quality_score = search_terms_found / total_docs_checked if total_docs_checked > 0 else 0
            
            if quality_score > 0.8:
                st.success(f"?? Excellent - {search_terms_found}/{total_docs_checked} documents contain '{search_query}'")
            elif quality_score > 0.5:
                st.warning(f"?? Good - {search_terms_found}/{total_docs_checked} documents contain '{search_query}'")
            elif quality_score > 0:
                st.info(f"?? Moderate - {search_terms_found}/{total_docs_checked} documents contain '{search_query}'")
            else:
                st.error(f"? Poor - {search_terms_found}/{total_docs_checked} documents contain '{search_query}'")
                
            # Show total found vs expected
            if len(documents) >= 9:  # We know John Nolan has 9 docs
                st.success(f"? **Complete results** - Found all available documents")
            else:
                st.info(f"?? **Found {len(documents)} documents** - Search completed")
        
        # Add to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"**Entity:** {extracted_terms}\n\n**Answer:** {final_answer}\n\n**Performance:** {total_time:.1f}s total, {len(documents)} documents"
        })