# main_app.py
# Production RAG System - Main Streamlit Application with Hybrid Search
# Final Version with Bug Fixes

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
    from utils.excel_export import render_excel_export_section
except ImportError as e:
    st.error(f"Import error: {e}")
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
    .hybrid-box {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 1px solid #dc3545;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    .database-method { background: #dc3545; color: white; }
    .hybrid-method { background: #fd7e14; color: white; }
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
    """Initialize the production RAG system with hybrid search"""
    try:
        logger.info("Initializing Production RAG System...")
        
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
            logger.warning(f"Some components failed to initialize: {failed_components}")
        
        logger.info("Production RAG System initialized successfully")
        
        return {
            "entity_extractor": entity_extractor,
            "query_rewriter": query_rewriter, 
            "retriever": retriever,
            "fusion_engine": fusion_engine,
            "status": component_status
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        logger.error(traceback.format_exc())
        raise

async def run_production_search(system_components: Dict, question: str):
    """Execute production-grade hybrid search pipeline"""
    
    # Set search in progress at the start
    st.session_state.search_in_progress = True
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    pipeline_start = time.time()
    
    try:
        # STAGE 1: Entity Extraction
        status_text.text("🧠 Smart entity extraction...")
        progress_bar.progress(15)
        
        extraction_start = time.time()
        entity_result = system_components["entity_extractor"].extract_entity(question)
        extraction_time = time.time() - extraction_start
        
        logger.info(f"Entity extraction: '{entity_result.entity}' via {entity_result.method} (confidence: {entity_result.confidence:.2f})")
        
        # STAGE 2: Query Rewriting  
        status_text.text("✏️ Query transformation...")
        progress_bar.progress(30)
        
        rewrite_start = time.time()
        rewrite_result = system_components["query_rewriter"].rewrite_query(
            question, entity_result.entity
        )
        rewrite_time = time.time() - rewrite_start
        
        logger.info(f"Query rewriting: {len(rewrite_result.rewrites)} variants via {rewrite_result.method}")
        
        # STAGE 3: 🆕 Hybrid Multi-Strategy Retrieval
        status_text.text("🔍 Hybrid multi-strategy retrieval...")
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
        
        # STAGE 4: 🆕 Hybrid Results Fusion
        status_text.text("⚖️ Advanced hybrid fusion...")
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
        status_text.text("📝 Generating intelligent answer...")
        progress_bar.progress(90)
        
        answer_start = time.time()
        answer = await generate_production_answer(
            question, fusion_result.fused_results, entity_result, rewrite_result
        )
        answer_time = time.time() - answer_start
        
        # Complete pipeline
        total_time = time.time() - pipeline_start
        
        progress_bar.progress(100)
        status_text.text("✅ Hybrid search completed!")
        
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
        logger.error(f"Production search failed: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Always clear search in progress flag
        st.session_state.search_in_progress = False

async def generate_production_answer(question: str, results: List[Any], entity_result: Any, rewrite_result: Any) -> str:
    """Generate production-quality answer with hybrid search context"""
    
    if not results:
        return f"""No relevant information found for your query: "{question}"

🔍 Search Summary:
- Entity extracted: "{entity_result.entity}" (confidence: {entity_result.confidence:.1%})
- Query variants tried: {len(rewrite_result.rewrites)}
- Method used: {entity_result.method}
- Search strategy: Hybrid (Vector + Database)

💡 Suggestions:
- Try rephrasing your query
- Use more specific terms
- Check if the information exists in the knowledge base"""

    # Analyze results quality and source distribution
    database_results = [r for r in results if "database" in r.source_method]
    vector_results = [r for r in results if ("vector" in r.source_method or "llamaindex" in r.source_method)]
    
    high_quality = [r for r in results if r.similarity_score >= 0.7]
    medium_quality = [r for r in results if 0.4 <= r.similarity_score < 0.7]
    
    # Generate contextual answer
    answer_parts = []
    
    # Header with hybrid search success
    answer_parts.append(f"🎯 Found **{len(results)} relevant documents** for **{entity_result.entity}**:")
    
    # Show source distribution
    if database_results and vector_results:
        answer_parts.append(f"📊 **Hybrid Search**: {len(database_results)} exact matches + {len(vector_results)} semantic matches")
    elif database_results:
        answer_parts.append(f"🗄️ **Database Search**: {len(database_results)} exact matches found")
    elif vector_results:
        answer_parts.append(f"🔍 **Vector Search**: {len(vector_results)} semantic matches found")
    
    # High quality results summary
    if high_quality:
        answer_parts.append(f"\n**📋 Primary Information** ({len(high_quality)} high-confidence documents):")
        for i, result in enumerate(high_quality[:3], 1):
            preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            source_indicator = "🗄️" if "database" in result.source_method else "🔍"
            answer_parts.append(f"{i}. {source_indicator} **{result.filename}** ({result.similarity_score:.3f}): {preview}")
    
    # Medium quality results summary  
    if medium_quality and len(high_quality) < 3:
        needed = 3 - len(high_quality)
        answer_parts.append(f"\n**📄 Additional Information** ({len(medium_quality)} medium-confidence documents):")
        for i, result in enumerate(medium_quality[:needed], len(high_quality) + 1):
            preview = result.content[:150] + "..." if len(result.content) > 150 else result.content
            source_indicator = "🗄️" if "database" in result.source_method else "🔍"
            answer_parts.append(f"{i}. {source_indicator} **{result.filename}** ({result.similarity_score:.3f}): {preview}")
    
    # Hybrid search intelligence summary
    answer_parts.append(f"\n**🧠 Search Intelligence:**")
    answer_parts.append(f"- Entity analysis: {entity_result.method} extraction")
    answer_parts.append(f"- Query variants: {len(rewrite_result.rewrites)} strategies tried")
    answer_parts.append(f"- Search approach: Hybrid (Database + Vector)")
    answer_parts.append(f"- Best match confidence: {max(r.similarity_score for r in results):.1%}")
    
    return "\n".join(answer_parts)

@st.cache_data(ttl=300)
def get_system_status():
    """Get cached system status with hybrid search info"""
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
            },
            "hybrid_enabled": config.search.enable_hybrid_search if hasattr(config.search, 'enable_hybrid_search') else True
        }
        
    except Exception as e:
        return {"error": str(e), "system": {}}

def render_sidebar():
    """Render enhanced sidebar with hybrid search status"""
    with st.sidebar:
        st.header("🔍 System Status")
        
        status = get_system_status()
        
        if "error" in status:
            st.error(f"System Error: {status['error']}")
            return status
        
        # Hybrid search indicator
        if status.get("hybrid_enabled", True):
            st.success("🚀 Hybrid Search Enabled")
        else:
            st.warning("⚠️ Vector-Only Mode")
        
        # Database status
        if status["database"]["available"]:
            st.success("🗄️ Database Connected")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", status["database"]["total_documents"])
            with col2:
                st.metric("Files", status["database"]["unique_files"])
        else:
            st.error("❌ Database Error")
        
        # Embedding status
        if status["embedding"]["available"]:
            st.success("🔍 Embeddings Ready")
            st.info(f"Model: {status['embedding']['model']}")
            st.info(f"Dimension: {status['embedding']['dimension']}")
        else:
            st.error("❌ Embedding Error")
        
        st.markdown("---")
        
        # Component status
        st.header("🔧 Components")
        
        components = status.get("components", {})
        
        if components.get("entity_extractors"):
            available_count = len([k for k, v in components['entity_extractors'].items() if v])
            st.success(f"🧠 Entity Extractors ({available_count})")
            for name, available in components["entity_extractors"].items():
                st.text(f"  {name}: {'✅' if available else '❌'}")
        
        if components.get("query_rewriters"):
            available_count = len([k for k, v in components['query_rewriters'].items() if v])
            st.success(f"✏️ Query Rewriters ({available_count})")
            for name, available in components["query_rewriters"].items():
                st.text(f"  {name}: {'✅' if available else '❌'}")
        
        if components.get("retrievers"):
            available_count = len([k for k, v in components['retrievers'].items() if v])
            st.success(f"🔄 Retrievers ({available_count})")
            for name, available in components["retrievers"].items():
                retriever_type = "🗄️" if "database" in name else "🔍"
                st.text(f"  {retriever_type} {name}: {'✅' if available else '❌'}")
        
        return status

def render_main_interface():
    """Render main interface with hybrid search branding"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Production RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">🚀 Hybrid Search • 🧠 Multi-Strategy Intelligence • ⚖️ Advanced Fusion • 📊 Excel Export Ready</p>', unsafe_allow_html=True)
    
    # Search interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        current_query = st.text_input(
            "Enter your question:",
            value=st.session_state.get("example_query", ""),
            placeholder="e.g., tell me about Breeda Daly (press Enter to search with hybrid approach)",
            key="main_query",
            on_change=on_query_change
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        
        # Button logic
        search_in_progress = st.session_state.get("search_in_progress", False)
        widget_value = st.session_state.get("main_query", "")
        return_value = current_query or ""
        final_query = return_value or widget_value
        
        has_actual_text = bool(final_query and final_query.strip() and len(final_query.strip()) > 0)
        search_disabled = (not has_actual_text) or search_in_progress
        
        # Create button with hybrid search indicator
        search_button = st.button(
            "🚀 Hybrid Search", 
            type="primary", 
            use_container_width=True,
            disabled=search_disabled,
            help="Combines database exact matching with vector semantic search"
        )
    
    return current_query or st.session_state.get("main_query", ""), search_button

def render_search_results(result: Dict):
    """Render comprehensive hybrid search results"""
    
    st.markdown("---")
    
    # Entity extraction results
    entity_result = result["entity_result"]
    st.markdown(f"""
    <div class="entity-box">
        <h4>🧠 Smart Entity Extraction</h4>
        <p><strong>Original:</strong> "{result['original_question']}"</p>
        <p><strong>Extracted:</strong> <span style="font-size: 1.2em; color: #007bff;"><strong>"{entity_result.entity}"</strong></span></p>
        <p><strong>Method:</strong> {entity_result.method} (confidence: {entity_result.confidence:.1%})</p>
        {f'<p><strong>Alternatives:</strong> {", ".join(entity_result.alternatives)}</p>' if entity_result.alternatives else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # Query rewriting results
    rewrite_result = result["rewrite_result"]
    with st.expander("✏️ Query Transformations", expanded=False):
        st.write(f"**Method:** {rewrite_result.method} (confidence: {rewrite_result.confidence:.1%})")
        st.write(f"**Generated {len(rewrite_result.rewrites)} variants:**")
        for i, variant in enumerate(rewrite_result.rewrites, 1):
            st.write(f"{i}. {variant}")
    
    # Main answer
    st.header("📋 Answer")
    st.markdown(f'<div class="success-box">{result["answer"]}</div>', unsafe_allow_html=True)
    
    # Performance metrics
    metrics = result["performance_metrics"]
    st.header("📊 Performance Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱️ Total Time", f"{metrics['total_time']:.2f}s")
    with col2:
        st.metric("📄 Results Found", result["fusion_result"].final_count)
    with col3:
        st.metric("🔧 Methods Used", len(result["retrieval_result"].methods_used))
    with col4:
        st.metric("⚖️ Fusion Method", result["fusion_result"].fusion_method.replace("_", " ").title())
    
    # Detailed metrics
    with st.expander("🔍 Pipeline Breakdown", expanded=False):
        efficiency = metrics["pipeline_efficiency"]
        st.write("**Time Distribution:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🧠 Entity Extraction", f"{metrics['extraction_time']:.3f}s", f"{efficiency['extraction_pct']:.1f}%")
            st.metric("✏️ Query Rewriting", f"{metrics['rewrite_time']:.3f}s", f"{efficiency['rewrite_pct']:.1f}%")
        
        with col2:
            st.metric("🔄 Multi-Retrieval", f"{metrics['retrieval_time']:.3f}s", f"{efficiency['retrieval_pct']:.1f}%")
            st.metric("⚖️ Results Fusion", f"{metrics['fusion_time']:.3f}s", f"{efficiency['fusion_pct']:.1f}%")
        
        with col3:
            st.metric("📝 Answer Generation", f"{metrics['answer_time']:.3f}s", f"{efficiency['answer_pct']:.1f}%")
            st.metric("🚀 Pipeline Efficiency", f"{(1/metrics['total_time']):.2f} q/s")
    
    # 🆕 Hybrid Retrieval Intelligence
    retrieval_result = result["retrieval_result"]
    st.header("🔍 Retrieval Intelligence")
    
    # Determine search strategy used
    methods_used = retrieval_result.methods_used
    is_hybrid = len([m for m in methods_used if "database" in m]) > 0 and len([m for m in methods_used if "vector" in m or "llamaindex" in m]) > 0
    
    if is_hybrid:
        st.markdown('<div class="hybrid-box"><h5>🚀 Hybrid Search Successfully Applied</h5><p>Combined database exact matching with vector semantic search for optimal results.</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Candidates", retrieval_result.total_candidates)
    with col2:
        st.metric("⚖️ After Fusion", result["fusion_result"].final_count)
    with col3:
        fusion_ratio = result["fusion_result"].final_count / retrieval_result.total_candidates if retrieval_result.total_candidates > 0 else 0
        st.metric("🎯 Quality Ratio", f"{fusion_ratio:.1%}")
    
    # Method badges with hybrid awareness
    st.write("**🔧 Methods Used:**")
    methods_html = ""
    for method in result["retrieval_result"].methods_used:
        if "database" in method:
            methods_html += f'<span class="method-badge database-method">🗄️ {method}</span>'
        elif "hybrid" in method:
            methods_html += f'<span class="method-badge hybrid-method">🚀 {method}</span>'
        elif "vector" in method or "llamaindex" in method:
            methods_html += f'<span class="method-badge primary-method">🔍 {method}</span>'
        else:
            methods_html += f'<span class="method-badge secondary-method">{method}</span>'
    
    st.markdown(methods_html, unsafe_allow_html=True)
    
    # Sources with hybrid source indicators
    if result["fusion_result"].fused_results:
        st.header(f"📁 Sources ({len(result['fusion_result'].fused_results)} documents)")
        
        # 🆕 Fixed quality distribution - use correct keys
        quality_dist = result["fusion_result"].fusion_metadata.get("quality_distribution", {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌟 Excellent", quality_dist.get("excellent", 0), "≥80% similarity")
        with col2:
            st.metric("✅ Good", quality_dist.get("good", 0), "≥60% similarity")
        with col3:
            st.metric("⚠️ Moderate", quality_dist.get("moderate", 0), "≥40% similarity")
        with col4:
            st.metric("📄 Low", quality_dist.get("low", 0), "<40% similarity")
        
        # Individual sources with hybrid indicators
        for i, doc in enumerate(result["fusion_result"].fused_results, 1):
            # Determine source type for icon
            source_icon = "🗄️" if "database" in doc.source_method else "🔍"
            source_label = "Database Match" if "database" in doc.source_method else "Vector Match"
            
            with st.expander(f"{source_icon} Document {i}. {doc.filename} (similarity: {doc.similarity_score:.3f})", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**📄 Content Preview:**")
                    st.text_area(
                        "preview", 
                        doc.content, 
                        height=150, 
                        key=f"preview_{i}", 
                        label_visibility="collapsed"
                    )
                
                with col2:
                    st.markdown("**🔍 Document Intelligence:**")
                    st.write(f"**Similarity:** {doc.similarity_score:.3f}")
                    st.write(f"**Source:** {source_label}")
                    st.write(f"**Method:** {doc.source_method}")
                    
                    # Show hybrid-specific metadata
                    if "database_strategy" in doc.metadata:
                        strategy = doc.metadata["database_strategy"]
                        st.write(f"**DB Strategy:** {strategy.replace('_', ' ').title()}")
                    
                    if "match_type" in doc.metadata:
                        match_type = doc.metadata["match_type"]
                        st.write(f"**Match Type:** {match_type.replace('_', ' ').title()}")
                    
                    if "query_occurrences" in doc.metadata:
                        occurrences = doc.metadata["query_occurrences"]
                        st.write(f"**Query Frequency:** {occurrences}x")
                    
                    # Show fusion scores
                    fusion_scores = []
                    for score_key in ["hybrid_weighted_score", "person_priority_score", "database_priority_score"]:
                        if score_key in doc.metadata:
                            fusion_scores.append(f"{score_key.replace('_', ' ').title()}: {doc.metadata[score_key]:.3f}")
                    
                    if fusion_scores:
                        st.write("**🎯 Fusion Scores:**")
                        for score in fusion_scores[:2]:  # Limit to 2 scores
                            st.write(f"  {score}")
                    
                    # Show hybrid fusion factors if available
                    if "fusion_factors" in doc.metadata:
                        factors = doc.metadata["fusion_factors"]
                        st.write("**⚖️ Quality Factors:**")
                        factor_indicators = {
                            "exact_query_match": "🎯",
                            "entity_match": "👤", 
                            "content_length_optimal": "📏",
                            "database_strategy": "🗄️"
                        }
                        
                        for factor, value in factors.items():
                            if factor in factor_indicators:
                                icon = factor_indicators[factor]
                                status = "✅" if value else "❌"
                                readable_name = factor.replace('_', ' ').title()
                                st.write(f"  {icon} {readable_name}: {status}")

def main():
    """Main application with hybrid search support"""
    
    init_session_state()
    
    # Initialize system
    try:
        if not st.session_state.system_initialized:
            system_components = initialize_production_system()
            st.session_state.system_components = system_components
            st.session_state.system_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        st.error("Please check your configuration and dependencies")
        st.stop()
    
    # Render sidebar
    status = render_sidebar()
    if "error" in status:
        st.stop()
    
    # Render main interface
    current_query, search_button = render_main_interface()
    
    # 🆕 Hybrid Search Advanced Settings
    with st.expander("🔧 Hybrid Search Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            show_entity_extraction = st.checkbox("🧠 Show entity extraction", value=True)
            show_query_variants = st.checkbox("✏️ Show query variants", value=False)
            show_performance = st.checkbox("📊 Show performance metrics", value=True)
            show_fusion_details = st.checkbox("⚖️ Show fusion details", value=True)
        with col2:
            show_retrieval_details = st.checkbox("🔍 Show retrieval intelligence", value=True)  
            show_source_breakdown = st.checkbox("📁 Show source breakdown", value=True)
            show_hybrid_indicators = st.checkbox("🚀 Show hybrid indicators", value=True)
            show_debug_info = st.checkbox("🐛 Show debug information", value=False)
        
        # Hybrid search strategy info
        if status.get("hybrid_enabled", True):
            st.info("🚀 **Hybrid Search Active**: Combines database exact matching with vector semantic search for optimal precision and recall.")
        else:
            st.warning("⚠️ **Vector-Only Mode**: Using traditional vector search. Enable hybrid search for better results on person queries.")
    
    # Handle search
    auto_search = st.session_state.get("auto_search_triggered", False)
    if auto_search:
        st.session_state.auto_search_triggered = False
        search_button = True
    
    if search_button and current_query.strip():
        
        # Always execute search when explicitly requested
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
            st.error(f"Hybrid search failed: {e}")
            logger.error(f"Search error: {e}")
            logger.error(traceback.format_exc())
            result = None
        
        finally:
            # Ensure search_in_progress is always reset
            st.session_state.search_in_progress = False
            st.rerun()
    
    # Display results
    if st.session_state.search_performed and st.session_state.search_results:
        render_search_results(st.session_state.search_results)
        
        # Add Excel export section
        render_excel_export_section(st.session_state.search_results)
        
        # Clear results button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.search_performed = False
                st.session_state.search_results = None
                st.session_state.last_query = ""
                st.rerun()
    
    # Footer with hybrid search branding
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 1rem; margin-top: 2rem;">
        <strong>🔍 Production RAG System</strong><br>
        🚀 Hybrid Search • 🧠 Multi-Strategy Intelligence • ⚖️ Advanced Fusion • 📊 Excel Export Ready<br>
        <small>Powered by LlamaIndex, Ollama & Streamlit | Enhanced with Database + Vector Search</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Application error: {e}")
        st.error("Please check the logs for more details")
