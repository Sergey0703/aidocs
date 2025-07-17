#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent RAG Test - Fixed UTF-8 Encoding
Testing dynamic top_k and content filter WITHOUT Streamlit
"""

import os
import time
import logging
import psycopg2
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "connection_string": os.getenv("SUPABASE_CONNECTION_STRING"),
    "embed_model": "mxbai-embed-large",
    "embed_dim": 1024,
    "ollama_url": "http://localhost:11434",
    "threshold": 0.35,     # Slightly higher threshold
    "max_top_k": 1000      # Safety limit for top_k
}

def get_optimal_top_k():
    """Get optimal top_k based on total documents in database"""
    try:
        conn = psycopg2.connect(CONFIG["connection_string"])
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM vecs.documents")
        total_docs = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        # Add buffer for safety and limit to max_top_k
        optimal_top_k = min(total_docs + 50, CONFIG["max_top_k"])
        logger.info(f"?? Database has {total_docs} documents, using top_k={optimal_top_k}")
        
        return optimal_top_k, total_docs
    except Exception as e:
        logger.error(f"? Error getting document count: {e}")
        # Fallback to reasonable default
        return 500, 450

def get_vector_components():
    """Initialize vector components"""
    logger.info("?? Initializing vector components...")
    
    try:
        vector_store = SupabaseVectorStore(
            postgres_connection_string=CONFIG["connection_string"],
            collection_name="documents",
            dimension=CONFIG["embed_dim"],
        )
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        embed_model = OllamaEmbedding(
            model_name=CONFIG["embed_model"], 
            base_url=CONFIG["ollama_url"]
        )
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        logger.info("? Vector components initialized successfully")
        return index, embed_model
        
    except Exception as e:
        logger.error(f"? Error initializing vector components: {e}")
        return None, None

def vector_search_with_filters(query, threshold=None):
    """Vector search with dynamic top_k + similarity filter + content filter"""
    
    logger.info(f"?? Vector search with filters for: '{query}'")
    start_time = time.time()
    
    try:
        # Initialize components
        index, embed_model = get_vector_components()
        if not index or not embed_model:
            return None
        
        # Use custom threshold if provided, otherwise use config
        threshold = threshold if threshold is not None else CONFIG["threshold"]
        
        # Get optimal top_k based on database size
        optimal_top_k, total_docs = get_optimal_top_k()
        
        # Create retriever with dynamic top_k
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=optimal_top_k,
            embed_model=embed_model
        )
        
        # Create similarity filter
        similarity_filter = SimilarityPostprocessor(similarity_cutoff=threshold)
        
        # Execute search
        logger.info(f"   ?? Searching in {total_docs} documents with top_k={optimal_top_k}")
        nodes = retriever.retrieve(query)
        logger.info(f"   Retrieved {len(nodes)} candidate nodes")
        
        # STEP 1: Apply similarity threshold filtering
        filtered_nodes = similarity_filter.postprocess_nodes(nodes)
        logger.info(f"   After similarity filter (={threshold}): {len(filtered_nodes)} nodes")
        
        # STEP 2: Apply content filter - only nodes that contain the search query
        final_nodes = [
            node for node in filtered_nodes
            if query.lower() in node.get_content().lower()
        ]
        logger.info(f"   After content filter (contains '{query}'): {len(final_nodes)} nodes")
        
        # Analysis
        total_time = time.time() - start_time
        
        # Calculate precision
        precision = (len(final_nodes) / len(filtered_nodes) * 100) if filtered_nodes else 0
        
        # Show detailed results
        logger.info(f"   ?? FILTERING RESULTS:")
        logger.info(f"      Total candidates: {len(nodes)}")
        logger.info(f"      After similarity: {len(filtered_nodes)}")
        logger.info(f"      After content: {len(final_nodes)}")
        logger.info(f"      Precision: {precision:.1f}%")
        logger.info(f"      Time: {total_time:.3f}s")
        
        # Show top 5 results
        if final_nodes:
            logger.info(f"   ?? Top 5 results:")
            for i, node in enumerate(final_nodes[:5], 1):
                similarity = getattr(node, 'score', 0.0)
                preview = node.get_content()[:80].replace('\n', ' ')
                
                # Try to get file name
                file_name = 'Unknown'
                try:
                    if hasattr(node, 'metadata') and isinstance(node.metadata, dict):
                        file_name = node.metadata.get('file_name', 'Unknown')
                except:
                    pass
                
                logger.info(f"      {i}. {similarity:.3f} | {file_name}")
                logger.info(f"         Preview: {preview}...")
        
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
            'final_nodes': final_nodes
        }
        
    except Exception as e:
        logger.error(f"? Error in vector search: {e}")
        return None

def test_different_thresholds():
    """Test different thresholds to find optimal values"""
    
    print("?? Testing Different Thresholds")
    print("=" * 60)
    
    # Check environment
    if not CONFIG["connection_string"]:
        print("? SUPABASE_CONNECTION_STRING not found!")
        return
    
    print("? Environment OK")
    
    test_entities = ["John Nolan", "Breeda Daly"]
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
    
    for entity in test_entities:
        print(f"\n?? Testing entity: {entity}")
        print("-" * 40)
        
        results = []
        for threshold in thresholds:
            print(f"\n?? Testing threshold: {threshold}")
            result = vector_search_with_filters(entity, threshold)
            if result:
                results.append(result)
        
        # Summary for this entity
        if results:
            print(f"\n?? SUMMARY for {entity}:")
            print("Threshold | Candidates | After Sim | After Content | Precision | Time")
            print("-" * 70)
            for r in results:
                print(f"   {r['threshold']:.2f}   |    {r['candidates']:3d}     |    {r['after_similarity']:3d}    |      {r['after_content']:3d}      |   {r['precision']:5.1f}%  | {r['time']:.2f}s")
            
            # Find best threshold
            best_result = max(results, key=lambda x: x['after_content'] if x['after_content'] > 0 else 0)
            print(f"\n?? Best threshold for {entity}: {best_result['threshold']:.2f}")
            print(f"   Found {best_result['after_content']} documents with 100% precision")
        
        print(f"\n" + "="*60)

def compare_old_vs_new():
    """Compare old approach (top_k=100) vs new approach (dynamic top_k + filters)"""
    
    print("?? Comparing Old vs New Approach")
    print("=" * 60)
    
    test_entities = ["John Nolan", "Breeda Daly"]
    
    for entity in test_entities:
        print(f"\n?? Testing entity: {entity}")
        print("-" * 40)
        
        # Old approach simulation (top_k=100, threshold=0.3, no content filter)
        print(f"\n?? OLD APPROACH (top_k=100, threshold=0.3, no content filter):")
        old_result = vector_search_old_style(entity)
        
        # New approach (dynamic top_k, threshold=0.35, content filter)
        print(f"\n?? NEW APPROACH (dynamic top_k, threshold=0.35, content filter):")
        new_result = vector_search_with_filters(entity, 0.35)
        
        # Comparison
        if old_result and new_result:
            print(f"\n?? COMPARISON for {entity}:")
            print("Approach | Top K | Threshold | Total Results | With Name | Precision")
            print("-" * 70)
            print(f"Old      | 100   |    0.30   |      {old_result['total_results']:3d}      |    {old_result['with_name']:3d}    |   {old_result['precision']:5.1f}%")
            print(f"New      | {new_result['optimal_top_k']:3d}   |    0.35   |      {new_result['after_content']:3d}      |    {new_result['after_content']:3d}    |   100.0%")
            
            improvement = new_result['after_content'] - old_result['with_name']
            print(f"\n?? Improvement: {improvement:+d} documents, precision: {100 - old_result['precision']:+.1f}%")
        
        print(f"\n" + "="*60)

def vector_search_old_style(query):
    """Simulate old style vector search for comparison"""
    
    logger.info(f"?? [OLD STYLE] Vector search for: '{query}'")
    start_time = time.time()
    
    try:
        # Initialize components
        index, embed_model = get_vector_components()
        if not index or not embed_model:
            return None
        
        # Old approach: fixed top_k=100, threshold=0.3
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=100,
            embed_model=embed_model
        )
        
        similarity_filter = SimilarityPostprocessor(similarity_cutoff=0.3)
        
        # Execute search
        nodes = retriever.retrieve(query)
        filtered_nodes = similarity_filter.postprocess_nodes(nodes)
        
        # Count nodes with query terms (but don't filter them out)
        nodes_with_query = sum(1 for node in filtered_nodes if query.lower() in node.get_content().lower())
        
        total_time = time.time() - start_time
        precision = (nodes_with_query / len(filtered_nodes) * 100) if filtered_nodes else 0
        
        logger.info(f"   ?? Old style results:")
        logger.info(f"      Total results: {len(filtered_nodes)}")
        logger.info(f"      With query terms: {nodes_with_query}")
        logger.info(f"      Precision: {precision:.1f}%")
        logger.info(f"      Time: {total_time:.3f}s")
        
        return {
            'total_results': len(filtered_nodes),
            'with_name': nodes_with_query,
            'precision': precision,
            'time': total_time
        }
        
    except Exception as e:
        logger.error(f"? Error in old style search: {e}")
        return None

def main():
    """Main function"""
    
    print("?? Intelligent RAG Test - Dynamic Top-K and Content Filter")
    print("=" * 70)
    print("1. Test different thresholds")
    print("2. Compare old vs new approach")
    print("3. Quick test with default settings")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        test_different_thresholds()
    elif choice == "2":
        compare_old_vs_new()
    elif choice == "3":
        print("\n?? Quick test with default settings:")
        for entity in ["John Nolan", "Breeda Daly"]:
            result = vector_search_with_filters(entity)
            if result:
                print(f"\n? {entity}: Found {result['after_content']} documents (100% precision)")
    elif choice == "4":
        print("?? Goodbye!")
    else:
        print("? Invalid option")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n?? Interrupted by user")
    except Exception as e:
        print(f"? Error: {e}")