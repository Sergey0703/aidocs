# main_app.py
# Production RAG System - Main Streamlit Application

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import time
import logging
import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Production RAG System",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import project modules
try:
    from config.settings import config
    from query_processing.entity_extractor import ProductionEntityExtractor
    from query_processing.query_rewriter import ProductionQueryRewriter
    from retrieval.multi_retriever import MultiStrategyRetriever
    from retrieval.results_fusion import ResultsFusionEngine
except ImportError as e:
    st.error(f"? Import error: {e}")
    st.error("Make sure all required files are in place and dependencies are installed")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #28a745;
        color: #155724;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .entity-box {
        background: linear-gradient(135deg, #e7f3ff, #b3d9ff);
        border: 1px solid #007bff;
        color: #004085;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 1px solid #ffc107;
        color: #856404;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .method-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .primary-method { background: #007bff; color: white; }
    .secondary-method { background: #6c757d; color: white; }
    .fusion-method { background: #28a745; color: white; }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
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
    if "system_initialized" not in st.session_state:
        st.session_state.system_initialized = False

def on_query_change():
    """Callback when query changes"""
    if st.session_state.main_query and st.session_state.main_query.strip():
        st.session_state.auto_search_triggered = True

@st.cache_resource
def initialize_production_system():
    """Initialize the production RAG system"""
    try:
        logger.info("?? Initializing Production RAG System...")
        
        # Validate configuration
        validation_results = config.validate_config()
        invalid_configs = [k for k, v in validation_results.items() if not v]
        
        if invalid_configs:
            error_msg = f"Invalid configuration: {', '.join(invalid_configs)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Initialize components
        entity_extractor = ProductionEntityExtractor(config)
        query_rewriter = ProductionQueryRewriter(config)
        retriever = MultiStrategyRetriever(config)
        fusion_engine = ResultsFusionEngine(config)
        
        # Check component status
        component_status = {
            "entity_extractor": len(entity_extractor.get_available_extractors()) > 0,
            "query_rewriter": len(query_rewriter.get_rewriter_status()) > 0,
            "retriever": len(retriever.get_retriever_status()) > 0,
            "fusion_engine": True
        }
        
        failed_components = [k for k, v in component_status.items() if not v]
        if failed_components:
            logger.warning(f"?? Some components failed to initialize: {failed_components}")
        
        logger.info("? Production RAG System initialized successfully")
        
        return {
            "entity_extractor": entity_extractor,
            "query_rewriter": query_rewriter, 
            "retriever": retriever,
            "fusion_engine": fusion_engine,
            "status": component_status
        }
        
    except Exception as e:
        logger.error(f"? Failed to initialize system: {e}")
        logger.error(traceback.format_exc())
        raise

async def run_production_search(system_components: Dict, question: str):
    """Execute production-grade search pipeline"""
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    pipeline_start = time.time()
    
    try:
        # STAGE 1: Entity Extraction
        status_text.text("?? Smart entity extraction...")
        progress_bar.progress(15)
        
        extraction_start = time.time()
        entity_result = system_components["entity_extractor"].extract_entity(question)
        extraction_time = time.time() - extraction_start
        
        logger.info(f"Entity extraction: '{entity_result.entity}' via {entity_result.method} (confidence: {entity_result.confidence:.2f})")
        
        # STAGE 2: Query Rewriting  
        status_text.text("?? Query transformation...")
        progress_bar.progress(30)
        
        rewrite_start = time.time()
        rewrite_result = system_components["query_rewriter"].rewrite_query(
            question, entity_result.entity
        )
        rewrite_time = time.time() - rewrite_start
        
        logger.info(f"Query rewriting: {len(rewrite_result.rewrites)} variants via {rewrite_result.method}")
        
        # STAGE 3: Multi-Strategy Retrieval
        status_text.text("?? Multi-strategy retrieval...")
        progress_bar.progress(50)
        
        retrieval_start = time.time()
        
        # Get required terms for content filtering
        required_terms = []
        if entity_result.entity != question.strip():
            # Extract words from entity for filtering
            entity_words = [word.lower() for word in entity_result.entity.split() 
                           if len(word) > 2 and word.lower() not in ['the', 'and', 'or']]
            required_terms = entity_words
        
        multi_retrieval_result = await system_components["retriever"].multi_retrieve(
            queries=rewrite_result.rewrites,
            extracted_entity=entity_result.entity,
            required_terms=required_terms
        )
        retrieval_time = time.time() - retrieval_start
        
        logger.info(f"Multi-retrieval: {len(multi_retrieval_result.results)} results via {', '.join(multi_retrieval_result.methods_used)}")
        
        # STAGE 4: Results Fusion
        status_text.text("?? Advanced results fusion...")
        progress_bar.progress(75)
        
        fusion_start = time.time()
        fusion_result = system_components["fusion_engine"].fuse_results(
            all_results=multi_retrieval_result.results,
            original_query=question,
            extracted_entity=entity_result.entity,
            required_terms=required_terms
        )
        fusion_time = time.time() - fusion_start
        
        logger.info(f"Results fusion: {fusion_result.final_count} final results via {fusion_result.fusion_method}")
        
        # STAGE 5: Answer Generation
        status_text.text("?? Generating intelligent answer...")
        progress_bar.progress(90)
        
        answer_start = time.time()
        answer = await generate_production_answer(
            question, fusion_result.fused_results, entity_result, rewrite_result
        )
        answer_time = time.time() - answer_start
        
        # Complete pipeline
        total_time = time.time() - pipeline_start
        
        progress_bar.progress(100)
        status_text.text("? Production search completed!")
        
        # Clear progress after delay
        await asyncio.sleep(1)
        progress_container.empty()
        
        return {
            "original_question": question,
            "entity_result": entity_result,
            "rewrite_result": rewrite_result,
            "retrieval_result": multi_retrieval_result,
            "fusion_result": fusion_result,
            "answer": answer,
            "performance_metrics": {
                "total_time": total_time,
                "extraction_time": extraction_time,
                "rewrite_time": rewrite_time,
                "retrieval_time": retrieval_time,
                "fusion_time": fusion_time,
                "answer_time": answer_time,
                "pipeline_efficiency": {
                    "extraction_pct": (extraction_time / total_time) * 100,
                    "rewrite_pct": (rewrite_time / total_time) * 100,
                    "retrieval_pct": (retrieval_time / total_time) * 100,
                    "fusion_pct": (fusion_time / total_time) * 100,
                    "answer_pct": (answer_time / total_time) * 100
                }
            }
        }
        
    except Exception as e:
        progress_container.empty()
        logger.error(f"? Production search failed: {e}")
        logger.error(traceback.format_exc())
        raise

async def generate_production_answer(question: str, results: List[Any], entity_result: Any, rewrite_result: Any) -> str:
    """Generate production-quality answer"""
    
    if not results:
        return f"""No relevant information found for your query: "{question}"

?? **Search Summary:**
- Entity extracted: "{entity_result.entity}" (confidence: {entity_result.confidence:.1%})
- Query variants tried: {len(rewrite_result.rewrites)}
- Method used: {entity_result.method}

?? **Suggestions:**
- Try rephrasing your query
- Use more specific terms
- Check if the information exists in the knowledge base"""

    # Analyze results quality
    high_quality = [r for r in results if r.similarity_score >= 0.7]
    medium_quality = [r for r in results if 0.4 <= r.similarity_score < 0.7]
    
    # Generate contextual answer
    answer_parts = []
    
    # Header with search success
    answer_parts.append(f"Found {len(results)} relevant documents for **{entity_result.entity}**:")
    
    # High quality results summary
    if high_quality:
        answer_parts.append(f"\n?? **Primary Information** ({len(high_quality)} documents):")
        for i, result in enumerate(high_quality[:3], 1):
            preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            answer_parts.append(f"{i}. **{result.filename}**: {preview}")
    
    # Medium quality results summary  
    if medium_quality and len(high_quality) < 3:
        needed = 3 - len(high_quality)
        answer_parts.append(f"\n?? **Additional Information** ({len(medium_quality)} documents):")
        for i, result in enumerate(medium_quality[:needed], len(high_quality) + 1):
            preview = result.content[:150] + "..." if len(result.content) > 150 else result.content
            answer_parts.append(f"{i}. **{result.filename}**: {preview}")
    
    # Search intelligence summary
    answer_parts.append(f"\n?? **Search Intelligence:**")
    answer_parts.append(f"- Query analysis: {entity_result.method} extraction")
    answer_parts.append(f"- Search variants: {len(rewrite_result.rewrites)} queries tried")
    answer_parts.append(f"- Best match confidence: {max(r.similarity_score for r in results):.1%}")
    
    return "\n".join(answer_parts)

@st.cache_data(ttl=300)
def get_system_status():
    """Get cached system status"""
    try:
        system = initialize_production_system()
        
        # Check database
        try:
            import psycopg2
            conn = psycopg2.connect(config.database.connection_string)
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {config.database.schema}.{config.database.table_name}")
            total_docs = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(DISTINCT metadata->>'file_name') FROM {config.database.schema}.{config.database.table_name} WHERE metadata->>'file_name' IS NOT NULL")
            unique_files = cur.fetchone()[0]
            cur.close()
            conn.close()
            
            database_status = {
                "available": True,
                "total_documents": total_docs,
                "unique_files": unique_files
            }
        except Exception as e:
            database_status = {"available": False, "error": str(e)}
        
        # Check embedding model
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
            embed_model = OllamaEmbedding(
                model_name=config.embedding.model_name,
                base_url=config.embedding.base_url
            )
            test_embedding = embed_model.get_text_embedding("test")
            embedding_status = {
                "available": len(test_embedding) == config.embedding.dimension,
                "model": config.embedding.model_name,
                "dimension": len(test_embedding) if test_embedding else 0
            }
        except Exception as e:
            embedding_status = {"available": False, "error": str(e)}
        
        return {
            "system": system["status"],
            "database": database_status,
            "embedding": embedding_status,
            "components": {
                "entity_extractors": system["entity_extractor"].get_extractor_status(),
                "query_rewriters": system["query_rewriter"].get_rewriter_status(),
                "retrievers": system["retriever"].get_retriever_status()
            }
        }
        
    except Exception as e:
        return {"error": str(e), "system": {}}

def render_sidebar():
    """Render enhanced sidebar"""
    with st.sidebar:
        st.header("?? System Status")
        
        status = get_system_status()
        
        if "error" in status:
            st.error(f"? System Error: {status['error']}")
            return status
        
        # Database status
        if status["database"]["available"]:
            st.success("? Database Connected")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", status["database"]["total_documents"])
            with col2:
                st.metric("Files", status["database"]["unique_files"])
        else:
            st.error("? Database Error")
        
        # Embedding status
        if status["embedding"]["available"]:
            st.success("? Embeddings Ready")
            st.info(f"Model: {status['embedding']['model']}")
            st.info(f"Dimension: {status['embedding']['dimension']}")
        else:
            st.error("? Embedding Error")
        
        st.markdown("---")
        
        # Component status
        st.header("?? Components")
        
        components = status.get("components", {})
        
        if components.get("entity_extractors"):
            st.success(f"? Entity Extractors ({len([k for k, v in components['entity_extractors'].items() if v])})")
            for name, available in components["entity_extractors"].items():
                icon = "?" if available else "?"
                st.text(f"  {icon} {name}")
        
        if components.get("query_rewriters"):
            st.success(f"? Query Rewriters ({len([k for k, v in components['query_rewriters'].items() if v])})")
            for name, available in components["query_rewriters"].items():
                icon = "?" if available else "?"
                st.text(f"  {icon} {name}")
        
        if components.get("retrievers"):
            st.success(f"? Retrievers ({len([k for k, v in components['retrievers'].items() if v])})")
            for name, available in components["retrievers"].items():
                icon = "?" if available else "?"
                st.text(f"  {icon} {name}")
        
        st.markdown("---")
        
        # Features
        st.header("? Production Features")
        features = [
            "?? Multi-method entity extraction",
            "?? Intelligent query rewriting", 
            "?? Multi-strategy retrieval",
            "?? Advanced results fusion",
            "?? Dynamic parameter tuning",
            "? Parallel processing",
            "?? Confidence scoring",
            "?? Performance analytics"
        ]
        
        for feature in features:
            st.markdown(f"- {feature}")
        
        st.markdown("---")
        
        # Example queries
        st.header("?? Test Queries")
        for example in config.ui.example_queries:
            if st.button(f"?? {example}", key=f"ex_{hash(example)}", use_container_width=True):
                st.session_state.example_query = example
        
        return status

def render_main_interface():
    """Render main interface"""
    
    # Header
    st.markdown('<h1 class="main-header">Production RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Multi-Strategy Intelligence • Advanced Fusion • Production Ready</p>', unsafe_allow_html=True)
    
    # Search interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        current_query = st.text_input(
            "Enter your question:",
            value=st.session_state.get("example_query", ""),
            placeholder="e.g., tell me about John Nolan (press Enter to search)",
            key="main_query",
            on_change=on_query_change
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        
        button_container = st.empty()
        
        if st.session_state.search_in_progress:
            button_container.empty()
            search_button = False
        else:
            search_disabled = not current_query.strip()
            search_button = button_container.button(
                "?? Search", 
                type="primary", 
                use_container_width=True,
                disabled=search_disabled
            )
    
    return current_query, search_button

def render_search_results(result: Dict):
    """Render comprehensive search results"""
    
    st.markdown("---")
    
    # Entity extraction results
    entity_result = result["entity_result"]
    st.markdown(f"""
    <div class="entity-box">
        <h4>?? Smart Entity Extraction</h4>
        <p><strong>Original:</strong> "{result['original_question']}"</p>
        <p><strong>Extracted:</strong> <span style="font-size: 1.2em; color: #007bff;"><strong>"{entity_result.entity}"</strong></span></p>
        <p><strong>Method:</strong> {entity_result.method} (confidence: {entity_result.confidence:.1%})</p>
        {f'<p><strong>Alternatives:</strong> {", ".join(entity_result.alternatives)}</p>' if entity_result.alternatives else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # Query rewriting results
    rewrite_result = result["rewrite_result"]
    with st.expander("?? Query Transformations", expanded=False):
        st.write(f"**Method:** {rewrite_result.method} (confidence: {rewrite_result.confidence:.1%})")
        st.write(f"**Generated {len(rewrite_result.rewrites)} variants:**")
        for i, variant in enumerate(rewrite_result.rewrites, 1):
            st.write(f"{i}. {variant}")
    
    # Main answer
    st.header("?? Answer")
    st.markdown(f'<div class="success-box">{result["answer"]}</div>', unsafe_allow_html=True)
    
    # Performance metrics
    metrics = result["performance_metrics"]
    st.header("?? Performance Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Time", f"{metrics['total_time']:.2f}s")
    with col2:
        st.metric("Results Found", result["fusion_result"].final_count)
    with col3:
        st.metric("Methods Used", len(result["retrieval_result"].methods_used))
    with col4:
        st.metric("Fusion Method", result["fusion_result"].fusion_method)
    
    # Detailed metrics
    with st.expander("?? Pipeline Breakdown", expanded=False):
        efficiency = metrics["pipeline_efficiency"]
        st.write("**Time Distribution:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Entity Extraction", f"{metrics['extraction_time']:.3f}s", f"{efficiency['extraction_pct']:.1f}%")
            st.metric("Query Rewriting", f"{metrics['rewrite_time']:.3f}s", f"{efficiency['rewrite_pct']:.1f}%")
        
        with col2:
            st.metric("Multi-Retrieval", f"{metrics['retrieval_time']:.3f}s", f"{efficiency['retrieval_pct']:.1f}%")
            st.metric("Results Fusion", f"{metrics['fusion_time']:.3f}s", f"{efficiency['fusion_pct']:.1f}%")
        
        with col3:
            st.metric("Answer Generation", f"{metrics['answer_time']:.3f}s", f"{efficiency['answer_pct']:.1f}%")
            st.metric("Pipeline Efficiency", f"{(1/metrics['total_time']):.2f} q/s")
    
    # Retrieval details  
    retrieval_result = result["retrieval_result"]
    st.header("?? Retrieval Intelligence")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Candidates", retrieval_result.total_candidates)
    with col2:
        st.metric("After Fusion", result["fusion_result"].final_count)
    with col3:
        fusion_ratio = result["fusion_result"].final_count / retrieval_result.total_candidates if retrieval_result.total_candidates > 0 else 0
        st.metric("Quality Ratio", f"{fusion_ratio:.1%}")
    
    # Method badges
    st.write("**Methods Used:**")
    methods_html = ""
    for method in result["retrieval_result"].methods_used:
        if "primary" in method:
            methods_html += f'<span class="method-badge primary-method">{method}</span>'
        elif "fusion" in method or "hybrid" in method:
            methods_html += f'<span class="method-badge fusion-method">{method}</span>'
        else:
            methods_html += f'<span class="method-badge secondary-method">{method}</span>'
    
    st.markdown(methods_html, unsafe_allow_html=True)
    
    # Sources
    if result["fusion_result"].fused_results:
        st.header(f"?? Sources ({len(result['fusion_result'].fused_results)} documents)")
        
        # Quality breakdown
        quality_dist = result["fusion_result"].fusion_metadata["quality_distribution"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High Quality", quality_dist["high"], "=70% similarity")
        with col2:
            st.metric("Medium Quality", quality_dist["medium"], "40-70% similarity")
        with col3:
            st.metric("Baseline", quality_dist["low"], "<40% similarity")
        
        # Individual sources
        for i, doc in enumerate(result["fusion_result"].fused_results, 1):
            quality_color = "#28a745" if doc.similarity_score >= 0.7 else "#ffc107" if doc.similarity_score >= 0.4 else "#dc3545"
            
            with st.expander(f"?? {i}. {doc.filename} (similarity: {doc.similarity_score:.3f})", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Content Preview:**")
                    st.text_area(
                        "preview", 
                        doc.content, 
                        height=150, 
                        key=f"preview_{i}", 
                        label_visibility="collapsed"
                    )
                
                with col2:
                    st.markdown("**Document Intelligence:**")
                    st.write(f"**Similarity:** {doc.similarity_score:.3f}")
                    st.write(f"**Source Method:** {doc.source_method}")
                    
                    if "weighted_score" in doc.metadata:
                        st.write(f"**Weighted Score:** {doc.metadata['weighted_score']:.3f}")
                    
                    if "fusion_factors" in doc.metadata:
                        factors = doc.metadata["fusion_factors"]
                        st.write("**Quality Factors:**")
                        for factor, value in factors.items():
                            icon = "?" if value else "?"
                            st.write(f"  {icon} {factor}")

def main():
    """Main application"""
    
    init_session_state()
    
    # Initialize system
    try:
        if not st.session_state.system_initialized:
            system_components = initialize_production_system()
            st.session_state.system_components = system_components
            st.session_state.system_initialized = True
    except Exception as e:
        st.error(f"? Failed to initialize system: {e}")
        st.error("Please check your configuration and dependencies")
        st.stop()
    
    # Render sidebar
    status = render_sidebar()
    if "error" in status:
        st.stop()
    
    # Render main interface
    current_query, search_button = render_main_interface()
    
    # Advanced settings
    with st.expander("?? Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            show_entity_extraction = st.checkbox("Show entity extraction", value=True)
            show_query_variants = st.checkbox("Show query variants", value=False)
            show_performance = st.checkbox("Show performance metrics", value=True)
        with col2:
            show_retrieval_details = st.checkbox("Show retrieval details", value=True)  
            show_fusion_analysis = st.checkbox("Show fusion analysis", value=False)
            show_debug_info = st.checkbox("Show debug information", value=False)
    
    # Handle search
    auto_search = st.session_state.get("auto_search_triggered", False)
    if auto_search:
        st.session_state.auto_search_triggered = False
        search_button = True
    
    if search_button and current_query.strip():
        
        # Check for cached results
        if current_query.strip() == st.session_state.last_query and st.session_state.search_results:
            st.info("?? Showing cached results for the same query")
        else:
            # Execute search
            st.session_state.search_in_progress = True
            st.session_state.last_query = current_query.strip()
            
            if "example_query" in st.session_state:
                st.session_state.example_query = ""
            
            try:
                result = asyncio.run(run_production_search(
                    st.session_state.system_components, 
                    current_query.strip()
                ))
                st.session_state.search_results = result
                st.session_state.search_performed = True
                
            except Exception as e:
                st.error(f"? Search failed: {e}")
                logger.error(f"Search error: {e}")
                logger.error(traceback.format_exc())
                result = None
            
            finally:
                st.session_state.search_in_progress = False
                st.rerun()
    
    # Display results
    if st.session_state.search_performed and st.session_state.search_results:
        render_search_results(st.session_state.search_results)
        
        # Clear results button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("?? Clear Results", use_container_width=True):
                st.session_state.search_performed = False
                st.session_state.search_results = None
                st.session_state.last_query = ""
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 1rem; margin-top: 2rem;">
        <strong>Production RAG System</strong><br>
        Multi-Strategy Intelligence • Advanced Fusion • Enterprise Ready<br>
        <small>Powered by LlamaIndex, Ollama & Streamlit</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        st.error(f"? Application error: {e}")
        st.error("Please check the logs for more details")