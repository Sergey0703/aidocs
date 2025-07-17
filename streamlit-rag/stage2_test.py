#!/usr/bin/env python3
"""
Vector Search Pagination Test
????????? ????????? ? ????? ??????? (3 ?????????) ??? ???????? ??????
"""

import os
import time
import logging
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import VectorIndexRetriever

load_dotenv()

# ????????? ???????????
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ????????????
CONFIG = {
    "connection_string": os.getenv("SUPABASE_CONNECTION_STRING"),
    "embed_model": "mxbai-embed-large",
    "embed_dim": 1024,
    "ollama_url": "http://localhost:11434",
    "test_batch_size": 3,    # ????????? ?????? ??? ????????????
    "real_batch_size": 100,  # ???????? ?????? ??? production
    "threshold": 0.3
}

def get_vector_components():
    """????????????? ????????? ??????????? (????????????????)"""
    
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
    
    return index, embed_model

def search_without_pagination(query, threshold, batch_size):
    """????? ??? ????????? (baseline ??? ?????????)"""
    
    logger.info(f"?? [NO PAGINATION] Searching '{query}' with batch_size={batch_size}, threshold={threshold}")
    start_time = time.time()
    
    index, embed_model = get_vector_components()
    
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=batch_size,
        embed_model=embed_model
    )
    
    # ???????? ??????????
    nodes = retriever.retrieve(query)
    
    # ????????? ?? threshold
    filtered_nodes = [node for node in nodes if getattr(node, 'score', 0.0) >= threshold]
    
    # ??????
    nodes_with_query = sum(1 for node in filtered_nodes if query.lower() in node.get_content().lower())
    similarities = [getattr(node, 'score', 0.0) for node in filtered_nodes]
    
    time_taken = time.time() - start_time
    
    logger.info(f"   Retrieved: {len(nodes)} total, {len(filtered_nodes)} after threshold")
    logger.info(f"   With query: {nodes_with_query}/{len(filtered_nodes)}")
    if similarities:
        logger.info(f"   Similarity range: {min(similarities):.3f}-{max(similarities):.3f}")
    logger.info(f"   Time: {time_taken:.3f}s")
    
    return {
        'method': 'no_pagination',
        'total_retrieved': len(nodes),
        'filtered_count': len(filtered_nodes),
        'with_query': nodes_with_query,
        'min_similarity': min(similarities) if similarities else 0,
        'max_similarity': max(similarities) if similarities else 0,
        'time': time_taken,
        'nodes': filtered_nodes
    }

def search_with_pagination(query, threshold, batch_size, max_pages=10):
    """????? ? ??????????"""
    
    logger.info(f"?? [WITH PAGINATION] Searching '{query}' with batch_size={batch_size}, threshold={threshold}")
    start_time = time.time()
    
    index, embed_model = get_vector_components()
    
    all_filtered_nodes = []
    all_retrieved_nodes = []
    page = 0
    
    while page < max_pages:
        page += 1
        page_start_time = time.time()
        
        logger.info(f"   ?? Page {page}: retrieving batch...")
        
        # ??????? ????? retriever ??? ?????? ????????
        # ??????????: LlamaIndex ????? ?? ???????????? offset ????????,
        # ??????? ?????????? top_k ? ??????? ?????? ? slice
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=batch_size * page,  # ???????? ??????, ????? ?????????? ??????????
            embed_model=embed_model
        )
        
        # ???????? ??????????
        nodes = retriever.retrieve(query)
        
        # ????? ?????? ????? ?????????? (?????????? ??? ????????????)
        start_idx = batch_size * (page - 1)
        end_idx = batch_size * page
        new_nodes = nodes[start_idx:end_idx] if start_idx < len(nodes) else []
        
        all_retrieved_nodes.extend(new_nodes)
        
        page_time = time.time() - page_start_time
        logger.info(f"   ?? Page {page}: got {len(new_nodes)} new nodes in {page_time:.3f}s")
        
        if not new_nodes:
            logger.info(f"   ?? Page {page}: no more results, stopping pagination")
            break
        
        # ?????????? similarity range ??? ???? ????????
        page_similarities = [getattr(node, 'score', 0.0) for node in new_nodes]
        if page_similarities:
            logger.info(f"   ?? Page {page}: similarity range {min(page_similarities):.3f}-{max(page_similarities):.3f}")
        
        # ????????? ?? threshold
        page_filtered = [node for node in new_nodes if getattr(node, 'score', 0.0) >= threshold]
        all_filtered_nodes.extend(page_filtered)
        
        logger.info(f"   ?? Page {page}: {len(page_filtered)} nodes passed threshold")
        
        # ????????? ??????? ?????????
        min_similarity = min(page_similarities) if page_similarities else 0
        if min_similarity < threshold:
            logger.info(f"   ?? Page {page}: min similarity ({min_similarity:.3f}) below threshold, stopping")
            break
        
        # ???? ???????? ?????? ??? batch_size, ?????? ????? ?? ?????
        if len(new_nodes) < batch_size:
            logger.info(f"   ?? Page {page}: got {len(new_nodes)} < {batch_size}, reached end")
            break
    
    # ?????? ????? ???????????
    nodes_with_query = sum(1 for node in all_filtered_nodes if query.lower() in node.get_content().lower())
    similarities = [getattr(node, 'score', 0.0) for node in all_filtered_nodes]
    
    time_taken = time.time() - start_time
    
    logger.info(f"   ?? PAGINATION SUMMARY:")
    logger.info(f"   Pages processed: {page}")
    logger.info(f"   Total retrieved: {len(all_retrieved_nodes)}, filtered: {len(all_filtered_nodes)}")
    logger.info(f"   With query: {nodes_with_query}/{len(all_filtered_nodes)}")
    if similarities:
        logger.info(f"   Similarity range: {min(similarities):.3f}-{max(similarities):.3f}")
    logger.info(f"   Total time: {time_taken:.3f}s")
    
    return {
        'method': 'with_pagination',
        'pages_processed': page,
        'total_retrieved': len(all_retrieved_nodes),
        'filtered_count': len(all_filtered_nodes),
        'with_query': nodes_with_query,
        'min_similarity': min(similarities) if similarities else 0,
        'max_similarity': max(similarities) if similarities else 0,
        'time': time_taken,
        'nodes': all_filtered_nodes
    }

def compare_pagination_vs_no_pagination(query):
    """?????????? ?????????? ? ?????????? ? ???"""
    
    logger.info(f"\n?? COMPARING PAGINATION vs NO PAGINATION for '{query}'")
    logger.info("=" * 70)
    
    # ???? 1: ??? ????????? (????????? batch)
    logger.info(f"\n?? Test 1: NO PAGINATION with small batch ({CONFIG['test_batch_size']})")
    no_pagination_small = search_without_pagination(
        query, CONFIG['threshold'], CONFIG['test_batch_size']
    )
    
    # ???? 2: ? ?????????? (????????? batch)
    logger.info(f"\n?? Test 2: WITH PAGINATION with small batch ({CONFIG['test_batch_size']})")
    with_pagination = search_with_pagination(
        query, CONFIG['threshold'], CONFIG['test_batch_size']
    )
    
    # ???? 3: ??? ????????? (??????? batch ??? reference)
    logger.info(f"\n?? Test 3: NO PAGINATION with large batch ({CONFIG['real_batch_size']})")
    no_pagination_large = search_without_pagination(
        query, CONFIG['threshold'], CONFIG['real_batch_size']
    )
    
    # ????????? ???????????
    logger.info(f"\n?? COMPARISON RESULTS:")
    logger.info("=" * 70)
    
    tests = [
        ("Small batch (no pagination)", no_pagination_small),
        ("Small batch (with pagination)", with_pagination),
        ("Large batch (reference)", no_pagination_large)
    ]
    
    logger.info(f"{'Method':<25} | {'Retrieved':<9} | {'Filtered':<8} | {'With Query':<10} | {'Time':<8}")
    logger.info("-" * 70)
    
    for name, result in tests:
        logger.info(f"{name:<25} | {result['total_retrieved']:>9} | {result['filtered_count']:>8} | "
                   f"{result['with_query']:>10} | {result['time']:>7.2f}s")
    
    # ?????? ????????????? ?????????
    logger.info(f"\n?? PAGINATION ANALYSIS:")
    
    # ?????????? ????????? ? reference (??????? batch)
    pagination_found = with_pagination['with_query']
    reference_found = no_pagination_large['with_query']
    
    if pagination_found == reference_found:
        logger.info(f"   ? PERFECT: Pagination found all {pagination_found} relevant documents!")
    elif pagination_found > reference_found:
        logger.info(f"   ?? UNEXPECTED: Pagination found MORE ({pagination_found}) than reference ({reference_found})")
    else:
        missed = reference_found - pagination_found
        logger.info(f"   ??  MISSED: Pagination found {pagination_found}, reference found {reference_found} (missed {missed})")
    
    # ????????????? ???????
    time_ratio = with_pagination['time'] / no_pagination_large['time']
    logger.info(f"   ??  Time efficiency: pagination took {time_ratio:.2f}x compared to large batch")
    
    # ????????? ???????????? ???????????
    pagination_files = set()
    reference_files = set()
    
    for node in with_pagination['nodes']:
        file_name = getattr(node, 'metadata', {}).get('file_name', 'Unknown') if hasattr(node, 'metadata') else str(hash(node.get_content()))
        pagination_files.add(file_name)
    
    for node in no_pagination_large['nodes']:
        file_name = getattr(node, 'metadata', {}).get('file_name', 'Unknown') if hasattr(node, 'metadata') else str(hash(node.get_content()))
        reference_files.add(file_name)
    
    missing_files = reference_files - pagination_files
    extra_files = pagination_files - reference_files
    
    if missing_files:
        logger.info(f"   ?? Files missed by pagination: {list(missing_files)[:3]}...")
    if extra_files:
        logger.info(f"   ?? Extra files found by pagination: {list(extra_files)[:3]}...")
    
    return {
        'no_pagination_small': no_pagination_small,
        'with_pagination': with_pagination,
        'no_pagination_large': no_pagination_large,
        'pagination_effective': pagination_found == reference_found
    }

def main():
    """??????? ??????? ????????????"""
    
    print("?? Vector Search Pagination Test")
    print("=" * 50)
    print(f"Test batch size: {CONFIG['test_batch_size']}")
    print(f"Real batch size: {CONFIG['real_batch_size']}")
    print(f"Threshold: {CONFIG['threshold']}")
    print("=" * 50)
    
    # ????????? ?????????
    if not CONFIG["connection_string"]:
        print("? SUPABASE_CONNECTION_STRING not found!")
        return
    
    print("? Environment OK")
    
    # ????????? ?? ????? entities
    test_entities = ["John Nolan", "Breeda Daly"]
    
    all_results = {}
    
    for entity in test_entities:
        print(f"\n{'='*70}")
        print(f"?? TESTING ENTITY: {entity}")
        print(f"{'='*70}")
        
        result = compare_pagination_vs_no_pagination(entity)
        all_results[entity] = result
    
    # ????????? ??????
    print(f"\n?? FINAL SUMMARY")
    print("=" * 50)
    
    all_effective = True
    for entity, result in all_results.items():
        effective = result['pagination_effective']
        all_effective = all_effective and effective
        status = "? EFFECTIVE" if effective else "? ISSUES"
        print(f"{entity}: {status}")
    
    if all_effective:
        print(f"\n?? PAGINATION WORKS PERFECTLY!")
        print(f"   Ready to use batch_size={CONFIG['real_batch_size']} in production")
    else:
        print(f"\n??  PAGINATION HAS ISSUES!")
        print(f"   Need to debug before production use")
    
    print("=" * 50)

if __name__ == "__main__":
    main()