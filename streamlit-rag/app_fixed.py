#!/usr/bin/env python3
"""
Fixed Streamlit RAG Chat Application
Corrected source extraction and display logic
"""

import os
import streamlit as st
import logging
import time
from typing import List, Dict
from dotenv import load_dotenv

try:
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.vector_stores.supabase import SupabaseVectorStore
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama
    from llama_index.core.postprocessor import SimilarityPostprocessor
except ImportError:
    # Fallback for different versions
    from llama_index import VectorStoreIndex, StorageContext
    from llama_index.vector_stores import SupabaseVectorStore
    from llama_index.embeddings import OllamaEmbedding
    from llama_index.llms import Ollama
    from llama_index.postprocessor import SimilarityPostprocessor

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chat System",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "connection_string": os.getenv("SUPABASE_CONNECTION_STRING"),
    "table_name": "documents",
    "embed_model": "nomic-embed-text",
    "chat_model": "llama3",
    "embed_dim": 768,
    "ollama_url": "http://localhost:11434"
}

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system (cached for performance)"""
    try:
        if not CONFIG["connection_string"]:
            st.error("? SUPABASE_CONNECTION_STRING not found in .env file!")
            st.stop()

        with st.spinner("?? Initializing RAG system..."):
            # Initialize components
            vector_store = SupabaseVectorStore(
                postgres_connection_string=CONFIG["connection_string"],
                collection_name=CONFIG["table_name"],
                dimension=CONFIG["embed_dim"],
            )
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            embed_model = OllamaEmbedding(
                model_name=CONFIG["embed_model"], 
                base_url=CONFIG["ollama_url"]
            )
            
            llm = Ollama(
                model=CONFIG["chat_model"],
                base_url=CONFIG["ollama_url"],
                temperature=st.session_state.get("temperature", 0.1),
                request_timeout=120.0
            )
            
            # Create index
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=embed_model
            )
            
            # Create chat engine with similarity threshold
            chat_engine = index.as_chat_engine(
                llm=llm,
                similarity_top_k=st.session_state.get("top_k", 5),
                chat_mode="context",
                # Add similarity cutoff to filter irrelevant results
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.4)
                ],
                system_prompt=st.session_state.get("system_prompt", 
                    "You are a helpful assistant that answers questions based on provided document context. "
                    "IMPORTANT: Only answer if the context contains relevant information about the specific question asked. "
                    "If the context doesn't contain relevant information, clearly state that the information is not available. "
                    "When answering, cite the specific document sources and be precise about what information comes from which source. "
                    "Do not make assumptions or provide information not explicitly contained in the context."
                )
            )
            
            return chat_engine, index
            
    except Exception as e:
        st.error(f"? Initialization error: {str(e)}")
        logger.error(f"RAG initialization error: {e}")
        st.stop()

def initialize_session_state():
    """Initialize session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "?? Hello! I'm ready to answer your questions about the uploaded documents. Ask me anything!",
                "sources": []
            }
        ]
    
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.1
    
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = (
            "You are a helpful assistant that answers questions based on provided document context. "
            "IMPORTANT: Only answer if the context contains relevant information about the specific question asked. "
            "If the context doesn't contain relevant information, clearly state that the information is not available. "
            "When answering, cite the specific document sources and be precise about what information comes from which source. "
            "Do not make assumptions or provide information not explicitly contained in the context."
        )

def extract_sources(response) -> List[Dict[str, str]]:
    """Extract sources from LlamaIndex response - CLEAN VERSION"""
    sources = []
    if hasattr(response, 'source_nodes') and response.source_nodes:
        for i, node in enumerate(response.source_nodes):
            if hasattr(node, 'metadata'):
                # Extract file identifiers
                file_name = (
                    node.metadata.get('file_name') or 
                    node.metadata.get('source') or 
                    node.metadata.get('filename') or
                    node.metadata.get('file_path') or
                    f"Document {node.node_id[:8]}"
                )
                
                # Extract content from the node
                content_preview = ""
                if hasattr(node, 'text') and node.text:
                    content_preview = node.text[:400] + "..." if len(node.text) > 400 else node.text
                elif hasattr(node, 'get_content'):
                    content = node.get_content()
                    content_preview = content[:400] + "..." if len(content) > 400 else content
                
                # Get relevance score
                score = getattr(node, 'score', 0)
                
                # Simple filtering - LlamaIndex should already filter by similarity_cutoff
                if content_preview.strip():
                    sources.append({
                        'file_name': file_name,
                        'content_preview': content_preview,
                        'metadata': node.metadata,
                        'node_id': node.node_id,
                        'score': score,
                        'file_path': node.metadata.get('source', file_name)
                    })
    
    # Remove duplicates and sort by relevance
    seen_files = set()
    unique_sources = []
    for source in sources:
        file_key = source['file_name']
        if file_key not in seen_files:
            seen_files.add(file_key)
            unique_sources.append(source)
    
    # Sort by relevance score (highest first)
    unique_sources.sort(key=lambda x: x['score'], reverse=True)
    
    return unique_sources

def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.header("?? Settings")
        
        # Model parameters
        st.subheader("?? LLM Parameters")
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.temperature,
            step=0.1,
            help="Controls response creativity. 0.0 = more precise, 1.0 = more creative"
        )
        
        top_k = st.slider(
            "Number of sources", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.top_k,
            help="How many documents to use for answering"
        )
        
        # System prompt
        st.subheader("?? System Prompt")
        system_prompt = st.text_area(
            "Assistant instructions:",
            value=st.session_state.system_prompt,
            height=100,
            help="Defines assistant behavior"
        )
        
        # Apply changes
        if st.button("?? Apply Settings"):
            st.session_state.temperature = temperature
            st.session_state.top_k = top_k
            st.session_state.system_prompt = system_prompt
            st.cache_resource.clear()
            st.rerun()
        
        st.divider()
        
        # Document Upload
        st.subheader("?? Document Upload")
        uploaded_file = st.file_uploader(
            "Upload new document:",
            type=['pdf', 'docx', 'txt'],
            help="Upload documents to add to the knowledge base"
        )
        
        if uploaded_file and st.button("?? Process Document"):
            st.success(f"Document {uploaded_file.name} would be processed")
            st.info("Feature coming soon: automatic document indexing")
        
        st.divider()
        
        # System information
        st.subheader("?? System Info")
        st.info(f"""
        **Chat Model:** {CONFIG['chat_model']}
        **Embedding Model:** {CONFIG['embed_model']}
        **Ollama URL:** {CONFIG['ollama_url']}
        **Table:** {CONFIG['table_name']}
        """)
        
        # Clear history
        if st.button("??? Clear History", type="secondary"):
            st.session_state.messages = [st.session_state.messages[0]]
            st.rerun()

def render_chat_message(message: Dict):
    """Render a single chat message"""
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"?? Sources ({len(message['sources'])})", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    if isinstance(source, dict):
                        # Color coding based on relevance
                        if source.get('score', 0) > 0.7:
                            relevance_color = "?? High"
                        elif source.get('score', 0) > 0.5:
                            relevance_color = "?? Medium"
                        else:
                            relevance_color = "?? Low"
                        
                        st.write(f"**{i}. {source['file_name']}** ({relevance_color})")
                        
                        # File info
                        file_path = source.get('file_path', source['file_name'])
                        if file_path and file_path != source['file_name']:
                            st.write(f"?? **File path:** `{file_path}`")
                        
                        # Content preview
                        if source.get('content_preview'):
                            st.write(f"**?? Content preview:**")
                            st.text_area(
                                f"Content from {source['file_name']}", 
                                value=source['content_preview'], 
                                height=120, 
                                key=f"hist_content_{i}_{len(st.session_state.messages)}_{hash(source['file_name'])}"
                            )
                        
                        if source.get('score'):
                            st.write(f"*Relevance Score:* {source['score']:.3f}")
                        
                        # Show metadata with checkbox
                        show_details = st.checkbox(
                            f"?? Show metadata for {source['file_name']}", 
                            key=f"hist_meta_{i}_{len(st.session_state.messages)}_{hash(source['file_name'])}"
                        )
                        
                        if show_details:
                            st.json(source['metadata'])
                        
                        st.divider()
                    else:
                        # Legacy string format
                        st.write(f"**{i}.** {source}")

def main():
    """Main application function"""
    # Header
    st.title("?? RAG Chat System")
    st.markdown("---")
    
    # Initialize
    initialize_session_state()
    render_sidebar()
    
    # Initialize RAG
    try:
        chat_engine, index = initialize_rag_system()
    except Exception as e:
        st.error("Failed to initialize system. Please check settings.")
        return
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display message history
        for message in st.session_state.messages:
            render_chat_message(message)
    
    # New message input
    if prompt := st.chat_input("?? Ask your question..."):
        # Add user message
        user_message = {"role": "user", "content": prompt, "sources": []}
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            try:
                # Show spinner during generation
                with st.spinner("?? Thinking..."):
                    start_time = time.time()
                    response = chat_engine.chat(prompt)
                    end_time = time.time()
                
                # Extract sources using fixed function
                sources = extract_sources(response)
                
                # Display response
                st.write(str(response.response))
                
                # Display sources with improved logic
                if sources:
                    with st.expander(f"?? Sources ({len(sources)}) - Relevance filtered", expanded=False):
                        for i, source in enumerate(sources, 1):
                            # Color coding based on relevance
                            if source['score'] > 0.7:
                                relevance_color = "?? High"
                            elif source['score'] > 0.5:
                                relevance_color = "?? Medium"
                            else:
                                relevance_color = "?? Low"
                            
                            st.write(f"**{i}. {source['file_name']}** ({relevance_color})")
                            
                            # File location info
                            file_path = source.get('file_path', source['file_name'])
                            if file_path and file_path != source['file_name']:
                                st.write(f"?? **File path:** `{file_path}`")
                                if st.button(f"?? Show file location", key=f"loc_{i}_{hash(source['file_name'])}"):
                                    full_path = f"/opt/rag_indexer/data/{file_path}" if not file_path.startswith('/') else file_path
                                    st.info(f"Full path: `{full_path}`")
                            
                            # Content preview
                            if source.get('content_preview'):
                                st.write(f"**?? Content preview:**")
                                st.text_area(
                                    f"Content from {source['file_name']}", 
                                    value=source['content_preview'], 
                                    height=120, 
                                    key=f"content_{i}_{hash(source['file_name'])}"
                                )
                            
                            if source.get('score'):
                                st.write(f"*Relevance Score:* {source['score']:.3f}")
                            
                            # Show metadata
                            show_details = st.checkbox(
                                f"?? Show metadata for {source['file_name']}", 
                                key=f"metadata_{i}_{hash(source['file_name'])}"
                            )
                            
                            if show_details:
                                st.json(source['metadata'])
                            
                            st.divider()
                
                # Show response time
                response_time = end_time - start_time
                st.caption(f"?? Response time: {response_time:.2f} sec")
                
                # Add response to history
                assistant_message = {
                    "role": "assistant", 
                    "content": str(response.response),
                    "sources": sources
                }
                st.session_state.messages.append(assistant_message)
                
            except Exception as e:
                error_msg = f"? Error generating response: {str(e)}"
                st.error(error_msg)
                logger.error(f"Chat error: {e}")
                
                # Add error message to history
                error_message = {
                    "role": "assistant", 
                    "content": "Sorry, an error occurred while processing your request. Please try again.",
                    "sources": []
                }
                st.session_state.messages.append(error_message)

if __name__ == "__main__":
    main()