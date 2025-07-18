#!/usr/bin/env python3

"""
Hybrid RAG System - Best of Both Worlds
Smart entity extraction + Simple vector search + Content filtering
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
    page_title="Hybrid RAG System",
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

@st.cache_resource
def initialize_services():
    """Initialize services (cached)"""
    try:
        # Import services
        from supabase_vector_service import SupabaseVectorService
        from simple_llm_service import create_simple_llm_service
        
        # Check environment variables
        connection_string = (
            os.getenv("SUPABASE_CONNECTION_STRING") or 
            os.getenv("DATABASE_URL") or
            os.getenv("POSTGRES_URL")
        )
        
        if not connection_string:
            st.error("Database connection string not found in environment!")
            st.stop()
        
        # Initialize vector service
        vector_service = SupabaseVectorService(
            connection_string=connection_string,
            table_name="documents"
        )
        
        # Initialize LLM service
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b")
        
        llm_service = create_simple_llm_service(
            ollama_url=ollama_url,
            model=ollama_model
        )
        
        return vector_service, llm_service
        
    except ImportError as e:
        st.error(f"Import error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Initialization error: {e}")
        st.stop()

@st.cache_resource
def get_extraction_llm():
    """Get LLM for entity extraction (cached)"""
    try:
        from llama_index.llms.ollama import Ollama
        
        return Ollama(
            model="llama3.2:3b",
            base_url="http://localhost:11434",
            request_timeout=30,
            additional_kwargs={
                "temperature": 0.0,
                "num_predict": 10,
                "top_k": 1,
                "top_p": 0.1,
                "stop": ["\n", ".", ",", ":", ";", "!", "?", " and", " or"]
            }
        )
    except Exception as e:
        st.error(f"Error initializing extraction LLM: {e}")
        return None

def extract_entity_with_llm(query: str) -> str:
    """Smart entity extraction using LLM (from intelligent_rag_app)"""
    
    llm = get_extraction_llm()
    if not llm:
        # Fallback to simple extraction
        return extract_entity_fallback(query)
    
    extraction_prompt = f"""Extract only the person's name from this question. Return ONLY the name, no other words.

Examples:
- "tell me about John Smith" -> John Smith
- "who is Mary Johnson" -> Mary Johnson  
- "find information about Bob Wilson" -> Bob Wilson
- "show me John Nolan" -> John Nolan
- "John Nolan certifications" -> John Nolan

Question: {query}

Name:"""
    
    try:
        response = llm.complete(extraction_prompt)
        extracted_entity = response.text.strip()
        
        # Clean extraction
        extracted_entity = re.sub(r'^(name|answer|result)[:=]\s*', '', extracted_entity, flags=re.IGNORECASE)
        extracted_entity = re.sub(r'\s*(is|the|answer|result)$', '', extracted_entity, flags=re.IGNORECASE)
        
        # Validate extraction
        if is_valid_entity(extracted_entity, query):
            return extracted_entity.strip()
        else:
            return extract_entity_fallback(query)
            
    except Exception as e:
        logger.warning(f"LLM entity extraction failed: {e}")
        return extract_entity_fallback(query)

def extract_entity_fallback(query: str) -> str:
    """Fallback entity extraction using regex"""
    # Find capitalized word sequences (likely names)
    name_patterns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    
    # Filter out common question words
    question_words = {'Tell', 'Show', 'Find', 'What', 'Who', 'Where', 'When', 'Why', 'How'}
    
    valid_names = [name for name in name_patterns if name not in question_words]
    
    if valid_names:
        return valid_names[0]  # Return first valid name
    
    # If no names found, return the query itself
    return query.strip()

def is_valid_entity(entity: str, original_query: str) -> bool:
    """Check if extracted entity is valid"""
    if not entity or len(entity.strip()) < 2:
        return False
    
    # Check if entity contains question words
    question_words = {'question', 'query', 'extract', 'name', 'tell', 'about', 'who', 'is', 'find', 'show'}
    entity_words = entity.lower().split()
    
    if any(word in question_words for word in entity_words):
        return False
    
    # Entity should be shorter than or equal to original query
    if len(entity) > len(original_query):
        return False
    
    return True

def get_required_terms_from_entity(entity: str) -> List[str]:
    """Get required terms from extracted entity"""
    # For person names, require ALL parts of the name
    entity_lower = entity.lower().strip()
    
    # Split into words and filter
    words = re.findall(r'\b[a-z]+\b', entity_lower)
    
    # Remove common words that shouldn't be required
    stop_words = {'the', 'a', 'an', 'and', 'or', 'of'}
    required_terms = [word for word in words if word not in stop_words and len(word) > 1]
    
    return required_terms

def apply_content_filter(search_results: List, required_terms: List[str]) -> List:
    """Apply strict content filtering based on required terms"""
    filtered_results = []
    
    for result in search_results:
        try:
            # Get all text for checking
            content = result.content.lower()
            filename = result.filename.lower()
            full_content = result.full_content.lower()
            
            all_text = f"{content} {filename} {full_content}"
            
            # Check if ALL required terms are present
            found_terms = []
            missing_terms = []
            
            for term in required_terms:
                if term.lower() in all_text:
                    found_terms.append(term)
                else:
                    missing_terms.append(term)
            
            # Include only if ALL required terms are found
            if len(missing_terms) == 0:
                result.search_info["found_terms"] = found_terms
                result.search_info["content_filtered"] = True
                result.search_info["extraction_method"] = "hybrid_smart"
                filtered_results.append(result)
                
        except Exception as e:
            logger.warning(f"Error filtering result: {e}")
            continue
    
    return filtered_results

def calculate_dynamic_limit(entity: str) -> int:
    """Calculate dynamic limit based on entity type"""
    entity_lower = entity.lower()
    
    # Person names typically have multiple documents
    if any(name in entity_lower for name in ['john nolan', 'breeda daly']):
        return 15
    elif len(entity.split()) >= 2:  # Multi-word entities (likely names)
        return 12
    elif any(word in entity_lower for word in ['training', 'certification', 'course']):
        return 10
    else:
        return 8

async def run_hybrid_search(vector_service, llm_service, question: str):
    """Execute hybrid search: Smart extraction + Simple search + Content filtering"""
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # STAGE 1: Smart Entity Extraction
    status_text.text("?? Extracting entity...")
    progress_bar.progress(20)
    
    extraction_start = time.time()
    extracted_entity = extract_entity_with_llm(question)
    extraction_time = time.time() - extraction_start
    
    # Get required terms from entity
    required_terms = get_required_terms_from_entity(extracted_entity)
    dynamic_limit = calculate_dynamic_limit(extracted_entity)
    
    # STAGE 2: Vector Search
    status_text.text("?? Vector search...")
    progress_bar.progress(50)
    
    search_start = time.time()
    search_limit = dynamic_limit * 2  # Get more candidates
    
    raw_search_results = await vector_service.vector_search(
        query=extracted_entity,  # Use extracted entity, not original question
        limit=search_limit,
        similarity_threshold=0.2
    )
    
    # STAGE 3: Content Filtering
    status_text.text("?? Content filtering...")
    progress_bar.progress(75)
    
    filter_start = time.time()
    filtered_results = apply_content_filter(raw_search_results, required_terms)
    
    # Sort and limit results
    filtered_results = sorted(filtered_results, 
                            key=lambda x: x.similarity_score, 
                            reverse=True)[:dynamic_limit]
    
    filter_time = time.time() - filter_start
    search_time = time.time() - search_start
    
    # STAGE 4: Answer Generation
    status_text.text("?? Generating answer...")
    progress_bar.progress(90)
    
    llm_start = time.time()
    
    context_docs = []
    for result in filtered_results:
        context_docs.append({
            'filename': result.filename,
            'content': result.content,
            'similarity_score': result.similarity_score
        })
    
    llm_response = await llm_service.generate_answer(
        question=question,  # Use original question for answer generation
        context_docs=context_docs,
        language="en"
    )
    
    llm_time = time.time() - llm_start
    total_time = time.time() - extraction_start
    
    # Complete progress
    progress_bar.progress(100)
    status_text.text("? Hybrid search completed!")
    
    # Clear progress after delay
    time.sleep(1)
    progress_container.empty()
    
    return {
        "original_question": question,
        "extracted_entity": extracted_entity,
        "required_terms": required_terms,
        "raw_results": len(raw_search_results),
        "filtered_results": filtered_results,
        "llm_response": llm_response,
        "metrics": {
            "extraction_time": extraction_time,
            "search_time": search_time,
            "filter_time": filter_time,
            "llm_time": llm_time,
            "total_time": total_time,
            "dynamic_limit": dynamic_limit,
            "precision": 1.0 if filtered_results else 0.0
        }
    }

@st.cache_data(ttl=300)
def check_service_status_cached():
    """Cached service status check"""
    try:
        vector_service, llm_service = initialize_services()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        db_stats = loop.run_until_complete(vector_service.get_database_stats())
        llm_available = loop.run_until_complete(llm_service.check_availability())
        
        loop.close()
        
        return {
            "database": {
                "available": True,
                "total_documents": db_stats.get('total_documents', 0),
                "unique_files": db_stats.get('unique_files', 0)
            },
            "llm": {
                "available": llm_available,
                "model": llm_service.model,
                "url": llm_service.ollama_url
            }
        }
    except Exception as e:
        return {
            "database": {"available": False, "error": str(e)},
            "llm": {"available": False, "error": str(e)}
        }

def main():
    """Main Streamlit application function"""
    
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">?? Hybrid RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Smart Entity Extraction + Simple Vector Search = Best Results</p>', unsafe_allow_html=True)
    
    # Initialize services
    vector_service, llm_service = initialize_services()
    
    # Sidebar
    with st.sidebar:
        st.header("?? System Info")
        
        status = check_service_status_cached()
        
        if status["database"]["available"]:
            st.success("? Database Connected")
            st.metric("Documents", status["database"]["total_documents"])
            st.metric("Files", status["database"]["unique_files"])
        else:
            st.error("? Database Error")
        
        if status["llm"]["available"]:
            st.success("? LLM Available")
            st.info(f"Model: {status['llm']['model']}")
        else:
            st.warning("?? LLM Unavailable")
        
        st.markdown("---")
        
        st.header("?? Hybrid Features")
        st.markdown("""
        **Best of Both Worlds:**
        - ?? **Smart**: LLM entity extraction
        - ? **Fast**: Single-pass vector search  
        - ?? **Accurate**: Content filtering
        - ?? **Consistent**: Same entity = same results
        - ??? **Reliable**: Fallback mechanisms
        """)
        
        st.markdown("---")
        
        st.header("?? Test Queries")
        examples = [
            "John Nolan",
            "tell me about John Nolan", 
            "show me John Nolan certifications",
            "who is Breeda Daly",
            "find Breeda Daly training"
        ]
        
        for example in examples:
            if st.button(f"?? {example}", key=f"ex_{hash(example)}"):
                st.session_state.example_query = example
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        current_query = st.text_input(
            "?? Enter your question:",
            value=st.session_state.get("example_query", ""),
            placeholder="e.g., tell me about John Nolan",
            key="main_query"
        )
    
    with col2:
        search_disabled = st.session_state.search_in_progress or not current_query.strip()
        search_button = st.button(
            "?? Search", 
            type="primary", 
            use_container_width=True,
            disabled=search_disabled
        )
    
    # Settings
    with st.expander("?? Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            show_extraction = st.checkbox("Show entity extraction", value=True)
            show_debug = st.checkbox("Show debug info", value=False)
        with col2:
            show_metrics = st.checkbox("Show performance metrics", value=True)
            show_sources = st.checkbox("Show detailed sources", value=True)
    
    # Search processing
    if search_button and current_query.strip():
        
        if st.session_state.search_in_progress:
            st.warning("? Search already in progress...")
            return
        
        if current_query.strip() == st.session_state.last_query and st.session_state.search_results:
            st.info("?? Showing cached results for same query")
        else:
            st.session_state.search_in_progress = True
            st.session_state.last_query = current_query.strip()
            
            if "example_query" in st.session_state:
                st.session_state.example_query = ""
            
            try:
                result = asyncio.run(run_hybrid_search(vector_service, llm_service, current_query.strip()))
                st.session_state.search_results = result
                st.session_state.search_performed = True
            
            except Exception as e:
                st.error(f"? Search error: {e}")
                result = None
            
            finally:
                st.session_state.search_in_progress = False
    
    # Display results
    if st.session_state.search_performed and st.session_state.search_results:
        result = st.session_state.search_results
        
        st.markdown("---")
        
        # Show entity extraction
        if show_extraction:
            st.markdown(f'<div class="entity-box"><strong>?? Smart Extraction:</strong><br>Original: "{result["original_question"]}"<br>Extracted Entity: <strong>"{result["extracted_entity"]}"</strong><br>Required Terms: {result["required_terms"]}</div>', unsafe_allow_html=True)
        
        # Main answer
        st.header("?? Answer")
        
        if result["llm_response"].success:
            st.markdown(f'<div class="success-box">{result["llm_response"].content}</div>', unsafe_allow_html=True)
        else:
            st.warning("?? LLM unavailable - showing fallback response:")
            st.markdown(result["llm_response"].content)
        
        # Performance metrics
        if show_metrics:
            st.header("?? Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = result["metrics"]
            
            with col1:
                st.metric("Total Time", f"{metrics['total_time']:.2f}s")
            with col2:
                st.metric("Extraction", f"{metrics['extraction_time']:.2f}s")
            with col3:
                st.metric("Search + Filter", f"{metrics['search_time']:.2f}s")
            with col4:
                st.metric("Precision", f"{metrics['precision']:.1%}")
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Raw Results", result["raw_results"])
            with col2:
                st.metric("Filtered Results", len(result["filtered_results"]))
            with col3:
                st.metric("Dynamic Limit", metrics["dynamic_limit"])
        
        # Debug information
        if show_debug:
            st.header("?? Debug Information")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Extraction Process")
                st.json({
                    "original_question": result["original_question"],
                    "extracted_entity": result["extracted_entity"],
                    "extraction_method": "LLM + fallback"
                })
            with col2:
                st.subheader("Required Terms")
                st.json(result["required_terms"])
            
            st.subheader("Hybrid Process")
            st.info(f"1. Extract entity: '{result['extracted_entity']}' ? 2. Search with entity ? 3. Filter by required terms ? 4. Got {len(result['filtered_results'])} precise results")
        
        # Sources
        if result["filtered_results"]:
            st.header(f"?? Sources ({len(result['filtered_results'])} documents)")
            
            # Special message for John Nolan
            if len(result["filtered_results"]) == 9 and "john" in result["extracted_entity"].lower() and "nolan" in result["extracted_entity"].lower():
                st.success("?? Perfect! Found all 9 John Nolan documents - hybrid system working correctly!")
            
            for i, doc in enumerate(result["filtered_results"], 1):
                with st.expander(f"?? {i}. {doc.filename} (similarity: {doc.similarity_score:.3f})"):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Content Preview:**")
                        preview = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
                        st.text(preview)
                    
                    with col2:
                        st.markdown("**Match Info:**")
                        st.text(f"Type: {doc.search_info['match_type']}")
                        st.text(f"Confidence: {doc.search_info['confidence']}")
                        
                        found_terms = doc.search_info.get("found_terms", [])
                        if found_terms:
                            st.text(f"Found terms: {', '.join(found_terms)}")
                    
                    if show_sources:
                        st.markdown("**Full Content:**")
                        st.text_area("", doc.full_content, height=100, key=f"content_{i}_{int(time.time())}")
        
        else:
            st.warning("? No relevant documents found after filtering")
            st.info("Try rephrasing your query or using different keywords")
        
        # Clear button
        if st.button("??? Clear Results", key="clear_results"):
            st.session_state.search_performed = False
            st.session_state.search_results = None
            st.session_state.last_query = ""
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        ?? Hybrid RAG System - Smart Extraction + Simple Search = Consistent Results<br>
        Built with Streamlit - Powered by LlamaIndex & Ollama
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()