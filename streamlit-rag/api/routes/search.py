# api/routes/search.py
# Search endpoint routes

import logging
import time
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict

from api.models.schemas import SearchRequest, SearchResponse, ErrorResponse
from api.core.dependencies import get_system_components, SystemComponents

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


async def execute_search(system_components: Dict, query: str):
    """Execute the search pipeline"""
    
    pipeline_start = time.time()
    
    try:
        # STAGE 1: Entity Extraction
        extraction_start = time.time()
        entity_result = system_components["entity_extractor"].extract_entity(query)
        extraction_time = time.time() - extraction_start
        
        logger.info(f"Entity extraction: '{entity_result.entity}' via {entity_result.method}")
        
        # STAGE 2: Query Rewriting
        rewrite_start = time.time()
        rewrite_result = system_components["query_rewriter"].rewrite_query(
            query, entity_result.entity
        )
        rewrite_time = time.time() - rewrite_start
        
        logger.info(f"Query rewriting: {len(rewrite_result.rewrites)} variants")
        
        # STAGE 3: Multi-Strategy Retrieval
        retrieval_start = time.time()
        
        required_terms = []
        if entity_result.entity != query.strip():
            entity_words = [word.lower() for word in entity_result.entity.split() 
                           if len(word) > 2 and word.lower() not in ['the', 'and', 'or']]
            required_terms = entity_words
        
        multi_retrieval_result = await system_components["retriever"].multi_retrieve(
            queries=rewrite_result.rewrites,
            extracted_entity=entity_result.entity,
            required_terms=required_terms
        )
        retrieval_time = time.time() - retrieval_start
        
        logger.info(f"Multi-retrieval: {len(multi_retrieval_result.results)} results")
        
        # STAGE 4: Results Fusion
        fusion_start = time.time()
        fusion_result = system_components["fusion_engine"].fuse_results(
            all_results=multi_retrieval_result.results,
            original_query=query,
            extracted_entity=entity_result.entity,
            required_terms=required_terms
        )
        fusion_time = time.time() - fusion_start
        
        logger.info(f"Results fusion: {fusion_result.final_count} final results")
        
        # STAGE 5: Generate Answer
        answer_start = time.time()
        answer = generate_answer(query, fusion_result.fused_results, entity_result)
        answer_time = time.time() - answer_start
        
        total_time = time.time() - pipeline_start
        
        return {
            "entity_result": entity_result,
            "rewrite_result": rewrite_result,
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
        logger.error(f"Search failed: {e}")
        raise


def generate_answer(query: str, results, entity_result) -> str:
    """Generate simple answer from results"""
    
    if not results:
        return f"No results found for: {query}"
    
    answer_parts = [
        f"Found {len(results)} documents for '{entity_result.entity}':",
        ""
    ]
    
    for i, result in enumerate(results[:5], 1):
        preview = result.content[:150] + "..." if len(result.content) > 150 else result.content
        answer_parts.append(f"{i}. {result.filename} (score: {result.similarity_score:.3f})")
        answer_parts.append(f"   {preview}")
        answer_parts.append("")
    
    return "\n".join(answer_parts)


@router.post("", response_model=SearchResponse, responses={500: {"model": ErrorResponse}})
async def search(
    request: SearchRequest,
    components: SystemComponents = Depends(get_system_components)
):
    """
    Execute hybrid search with entity extraction, query rewriting, and multi-strategy retrieval.
    
    - **query**: Search query text (1-1000 characters)
    - **max_results**: Maximum results to return (1-100, default: 20)
    - **similarity_threshold**: Optional similarity threshold (0.0-1.0)
    """
    
    try:
        system_components = components.get_components()
        
        result = await execute_search(system_components, request.query)
        
        # Convert to response model
        from api.models.schemas import (
            EntityResult, QueryRewriteResult, DocumentResult, PerformanceMetrics
        )
        
        return SearchResponse(
            success=True,
            query=request.query,
            entity_result=EntityResult(
                entity=result["entity_result"].entity,
                confidence=result["entity_result"].confidence,
                method=result["entity_result"].method,
                alternatives=result["entity_result"].alternatives,
                metadata=result["entity_result"].metadata
            ),
            rewrite_result=QueryRewriteResult(
                original_query=result["rewrite_result"].original_query,
                rewrites=result["rewrite_result"].rewrites,
                method=result["rewrite_result"].method,
                confidence=result["rewrite_result"].confidence,
                metadata=result["rewrite_result"].metadata
            ),
            results=[
                DocumentResult(
                    filename=doc.filename,
                    content=doc.content,
                    full_content=doc.full_content,
                    similarity_score=doc.similarity_score,
                    source_method=doc.source_method,
                    document_id=doc.document_id,
                    chunk_index=doc.chunk_index,
                    metadata=doc.metadata
                )
                for doc in result["fusion_result"].fused_results[:request.max_results]
            ],
            answer=result["answer"],
            total_results=len(result["fusion_result"].fused_results),
            performance_metrics=PerformanceMetrics(**result["performance_metrics"])
        )
        
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))