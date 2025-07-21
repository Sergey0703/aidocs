#!/usr/bin/env python3

"""
Simple RAG System - Using Working Console Script Logic in Streamlit
Direct port from intelligent_rag_test.py that we know works
"""

import streamlit as st
import time
import logging
import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Add import paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Simple RAG System",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .entity-box {
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
        color: #004085;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state only once"""
    if "search_performed" not in st.session_state:
        st.session_state.search_performed = False
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "search_in_progress" not in st.session_state:
        st.session_state.search_in_progress = False
    if "auto_search_triggered" not in st.session_state:
        st.session_state.auto_search_triggered = False

def on_query_change():
    """Callback when query text changes (Enter pressed)"""
    if st.session_state.main_query and st.session_state.main_query.strip():
        st.session_state.auto_search_triggered = True

@st.cache_resource
def get_vector_components():
    """Initialize vector components using the EXACT same code as console script"""
    logger.info("?? Initializing vector components...")
    
    try:
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.vector_stores.supabase import SupabaseVectorStore
        from llama_index.embeddings.ollama import OllamaEmbedding
        
        connection_string = os.getenv("SUPABASE_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("SUPABASE_CONNECTION_STRING not found!")
        
        vector_store = SupabaseVectorStore(
            postgres_connection_string=connection_string,
            collection_name="documents",
            dimension=768,  # nomic-embed-text dimension
        )
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text", 
            base_url="http://localhost:11434"
        )
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        logger.info("? Vector components initialized successfully (model: nomic-embed-text)")
        return index, embed_model
        
    except Exception as e:
        logger.error(f"? Error initializing vector components: {e}")
        st.error(f"Vector initialization error: {e}")
        return None, None

async def vector_search_console_logic(query, threshold=0.35):
    """Use the EXACT same logic as the working console script"""
    
    logger.info(f"?? Vector search for: '{query}' (threshold: {threshold})")
    start_time = time.time()
    
    try:
        # Initialize components (same as console script)
        index, embed_model = get_vector_components()
        if not index or not embed_model:
            return None
        
        # Import retriever components (same as console script)
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core.postprocessor import SimilarityPostprocessor
        
        # Get optimal top_k (same logic as console script)
        import psycopg2
        connection_string = os.getenv("SUPABASE_CONNECTION_STRING")
        conn = psycopg2.connect(connection_string)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM vecs.documents")
        total_docs = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        optimal_top_k = min(total_docs + 50, 1000)  # Same logic as console script
        logger.info(f"?? Database has {total_docs} documents, using top_k={optimal_top_k}")
        
        # Create retriever (same parameters as console script)
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=optimal_top_k,
            embed_model=embed_model
        )
        
        # Create similarity filter (same as console script)
        similarity_filter = SimilarityPostprocessor(similarity_cutoff=threshold)
        
        # Execute search (same as console script)
        logger.info(f"?? Searching in {total_docs} documents with top_k={optimal_top_k}")
        nodes = retriever.retrieve(query)
        logger.info(f"Retrieved {len(nodes)} candidate nodes")
        
        # Apply similarity threshold filtering (same as console script)
        filtered_nodes = similarity_filter.postprocess_nodes(nodes)
        logger.info(f"After similarity filter (={threshold}): {len(filtered_nodes)} nodes")
        
        # Apply content filter (same as console script)
        final_nodes = [
            node for node in filtered_nodes
            if query.lower() in node.get_content().lower()
        ]
        logger.info(f"After content filter (contains '{query}'): {len(final_nodes)} nodes")
        
        total_time = time.time() - start_time
        precision = (len(final_nodes) / len(filtered_nodes) * 100) if filtered_nodes else 0
        
        # Log results (same as console script)
        logger.info(f"?? FILTERING RESULTS:")
        logger.info(f"   Total candidates: {len(nodes)}")
        logger.info(f"   After similarity: {len(filtered_nodes)}")
        logger.info(f"   After content: {len(final_nodes)}")
        logger.info(f"   Precision: {precision:.1f}%")
        logger.info(f"   Time: {total_time:.3f}s")
        
        # Process nodes for display
        processed_nodes = []
        for i, node in enumerate(final_nodes):
            try:
                similarity = getattr(node, 'score', 0.0)
                content = node.get_content()
                
                # Get file name from metadata
                file_name = 'Unknown'
                if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                    metadata = node.node.metadata
                    if isinstance(metadata, dict):
                        file_name = metadata.get('file_name', 'Unknown')
                
                processed_nodes.append({
                    'filename': file_name,
                    'content': content,
                    'similarity_score': similarity,
                    'node': node
                })
                
            except Exception as e:
                logger.warning(f"Error processing node {i}: {e}")
                continue
        
        return {
            'query': query,
            'threshold': threshold,
            'total_docs': total_docs,
            'optimal_top_k': optimal_top_k,
            'candidates': len(nodes),
            'after_similarity': len(filtered_nodes),
            'after_content': len(final_nodes),
            'precision': precision,
            'time': total_time,
            'processed_nodes': processed_nodes
        }
        
    except Exception as e:
        logger.error(f"? Error in vector search: {e}")
        st.error(f"Search error: {e}")
        return None

def create_simple_answer(search_result):
    """Create simple answer from search results"""
    if not search_result or not search_result['processed_nodes']:
        return "No relevant documents found for your query."
    
    query = search_result['query']
    nodes = search_result['processed_nodes']
    
    # Simple answer generation
    answer_parts = [f"Found {len(nodes)} documents related to '{query}':\n"]
    
    for i, node in enumerate(nodes[:3], 1):  # Show top 3
        preview = node['content'][:200] + "..." if len(node['content']) > 200 else node['content']
        answer_parts.append(f"{i}. {node['filename']}: {preview}")
    
    return "\n\n".join(answer_parts)

async def run_simple_search(question: str):
    """Execute search using console script logic"""
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # STAGE 1: Simple entity extraction
    status_text.text("?? Processing query...")
    progress_bar.progress(25)
    
    # For simplicity, use query as-is
    extracted_entity = question.strip()
    
    # STAGE 2: Vector search using console logic
    status_text.text("?? Vector search...")
    progress_bar.progress(50)
    
    search_result = await vector_search_console_logic(extracted_entity, threshold=0.35)
    
    # STAGE 3: Create answer
    status_text.text("?? Creating answer...")
    progress_bar.progress(75)
    
    if search_result:
        answer = create_simple_answer(search_result)
    else:
        answer = "Search failed. Please check the logs for details."
    
    # Complete progress
    progress_bar.progress(100)
    status_text.text("? Search completed!")
    
    # Clear progress after delay
    time.sleep(1)
    progress_container.empty()
    
    return {
        "original_question": question,
        "extracted_entity": extracted_entity,
        "answer": answer,
        "search_result": search_result
    }

@st.cache_data(ttl=300)
def check_service_status():
    """Check service status"""
    try:
        connection_string = os.getenv("SUPABASE_CONNECTION_STRING")
        if not connection_string:
            return {"database": {"available": False, "error": "No connection string"}}
        
        import psycopg2
        conn = psycopg2.connect(connection_string)
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM vecs.documents")
        total_docs = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(DISTINCT metadata->>'file_name') FROM vecs.documents WHERE metadata->>'file_name' IS NOT NULL")
        unique_files = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        # Check Ollama
        ollama_available = True
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
            embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
            test_embedding = embed_model.get_text_embedding("test")
            ollama_available = len(test_embedding) > 0
        except:
            ollama_available = False
        
        return {
            "database": {
                "available": True,
                "total_documents": total_docs,
                "unique_files": unique_files
            },
            "ollama": {
                "available": ollama_available,
                "model": "nomic-embed-text"
            }
        }
        
    except Exception as e:
        return {
            "database": {"available": False, "error": str(e)},
            "ollama": {"available": False, "error": str(e)}
        }

def main():
    """Main Streamlit application"""
    
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">Simple RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Using Console Script Logic in Streamlit</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("System Status")
        
        status = check_service_status()
        
        if status["database"]["available"]:
            st.success("? Database Connected")
            st.metric("Documents", status["database"]["total_documents"])
            st.metric("Files", status["database"]["unique_files"])
        else:
            st.error(f"? Database Error: {status['database'].get('error', 'Unknown')}")
        
        if status["ollama"]["available"]:
            st.success("? Ollama Available")
            st.info(f"Model: {status['ollama']['model']}")
        else:
            st.error(f"? Ollama Error: {status['ollama'].get('error', 'Unknown')}")
        
        st.markdown("---")
        st.header("Test Queries")
        examples = ["John Nolan", "Breeda Daly", "tell me about John Nolan"]
        
        for example in examples:
            if st.button(f"?? {example}", key=f"ex_{hash(example)}"):
                st.session_state.example_query = example
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        current_query = st.text_input(
            "Enter your question:",
            value=st.session_state.get("example_query", ""),
            placeholder="e.g., John Nolan",
            key="main_query",
            on_change=on_query_change
        )
    
    with col2:
        st.markdown("")
        st.markdown("")
        
        button_container = st.empty()
        
        if st.session_state.search_in_progress:
            button_container.empty()
            search_button = False
        else:
            with button_container.container():
                search_disabled = not current_query.strip()
                search_button = st.button("Search", type="primary", use_container_width=True, disabled=search_disabled)
    
    # Auto-search trigger
    auto_search = st.session_state.get("auto_search_triggered", False)
    if auto_search:
        st.session_state.auto_search_triggered = False
        search_button = True
    
    # Search processing
    if search_button and current_query.strip():
        
        if current_query.strip() == st.session_state.last_query and st.session_state.search_results:
            st.info("?? Showing cached results")
        else:
            st.session_state.search_in_progress = True
            st.session_state.last_query = current_query.strip()
            
            if "example_query" in st.session_state:
                st.session_state.example_query = ""
            
            try:
                result = asyncio.run(run_simple_search(current_query.strip()))
                st.session_state.search_results = result
                st.session_state.search_performed = True
            
            except Exception as e:
                st.error(f"? Search error: {e}")
                result = None
            
            finally:
                st.session_state.search_in_progress = False
                st.rerun()
    
    # Display results
    if st.session_state.search_performed and st.session_state.search_results:
        result = st.session_state.search_results
        
        st.markdown("---")
        
        # Answer
        st.header("Answer")
        st.markdown(f'<div class="success-box">{result["answer"]}</div>', unsafe_allow_html=True)
        
        # Search details
        if result["search_result"]:
            search_data = result["search_result"]
            
            st.header("Search Details")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Candidates", search_data["candidates"])
            with col2:
                st.metric("After Similarity", search_data["after_similarity"])
            with col3:
                st.metric("Final Results", search_data["after_content"])
            with col4:
                st.metric("Precision", f"{search_data['precision']:.1f}%")
            
            # Sources
            if search_data["processed_nodes"]:
                st.header(f"Sources ({len(search_data['processed_nodes'])} found)")
                
                for i, node in enumerate(search_data["processed_nodes"], 1):
                    with st.expander(f"?? {i}. {node['filename']} (similarity: {node['similarity_score']:.3f})"):
                        preview = node['content'][:500] + "..." if len(node['content']) > 500 else node['content']
                        st.text(preview)
        
        # Clear button
        if st.button("Clear Results"):
            st.session_state.search_performed = False
            st.session_state.search_results = None
            st.session_state.last_query = ""
            st.rerun()

if __name__ == "__main__":
    main()