# retrieval/multi_retriever.py
# Multi-strategy retrieval system with DIAGNOSTIC LOGGING and timeout fixes

import time
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import concurrent.futures

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Single retrieval result"""
    content: str
    full_content: str
    filename: str
    similarity_score: float
    metadata: Dict[str, Any]
    source_method: str
    document_id: str = ""
    chunk_index: int = 0
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}

@dataclass
class MultiRetrievalResult:
    """Combined results from multiple retrieval strategies"""
    query: str
    results: List[RetrievalResult]
    methods_used: List[str]
    total_candidates: int
    retrieval_time: float
    fusion_method: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseRetriever(ABC):
    """Base class for retrievers"""
    
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[RetrievalResult]:
        """Retrieve documents for query"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if retriever is available"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get retriever name"""
        pass

class LlamaIndexRetriever(BaseRetriever):
    """LlamaIndex-based vector retriever with diagnostic logging"""
    
    def __init__(self, config):
        self.config = config
        self.index = None
        self.embed_model = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LlamaIndex components"""
        try:
            from llama_index.core import VectorStoreIndex, StorageContext
            from llama_index.vector_stores.supabase import SupabaseVectorStore
            from llama_index.embeddings.ollama import OllamaEmbedding
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core.postprocessor import SimilarityPostprocessor
            
            # Initialize embedding model
            self.embed_model = OllamaEmbedding(
                model_name=self.config.embedding.model_name,
                base_url=self.config.embedding.base_url
            )
            
            # Initialize vector store
            vector_store = SupabaseVectorStore(
                postgres_connection_string=self.config.database.connection_string,
                collection_name=self.config.database.table_name,
                dimension=self.config.embedding.dimension,
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            
            logger.info("? LlamaIndex Retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"? Failed to initialize LlamaIndex Retriever: {e}")
            self.index = None
            self.embed_model = None
    
    def is_available(self) -> bool:
        """Check if LlamaIndex components are available"""
        return self.index is not None and self.embed_model is not None
    
    def get_name(self) -> str:
        return "llamaindex_vector"
    
    async def retrieve(self, query: str, top_k: int = 10, similarity_threshold: float = 0.3, **kwargs) -> List[RetrievalResult]:
        """Retrieve using LlamaIndex with DETAILED DIAGNOSTIC LOGGING"""
        if not self.is_available():
            logger.warning("? LlamaIndex retriever not available")
            return []
        
        logger.info(f"?? DIAGNOSTIC: Starting LlamaIndex retrieval for '{query}' (threshold: {similarity_threshold}, top_k: {top_k})")
        
        try:
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core.postprocessor import SimilarityPostprocessor
            
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k * 2,  # Get more candidates for filtering
                embed_model=self.embed_model
            )
            
            # Create similarity postprocessor
            similarity_postprocessor = SimilarityPostprocessor(
                similarity_cutoff=similarity_threshold
            )
            
            # Retrieve nodes
            logger.info(f"?? Retrieving nodes with top_k={top_k * 2}")
            nodes = retriever.retrieve(query)
            logger.info(f"?? Retrieved {len(nodes)} raw candidate nodes")
            
            # Apply similarity filtering
            logger.info(f"?? Applying similarity filter (threshold: {similarity_threshold})")
            filtered_nodes = similarity_postprocessor.postprocess_nodes(nodes)
            logger.info(f"?? After similarity filtering: {len(filtered_nodes)} nodes")
            
            # Convert to RetrievalResult objects
            results = []
            query_lower = query.lower()
            
            logger.info(f"?? DETAILED ANALYSIS of top {min(len(filtered_nodes), top_k)} results:")
            
            for i, node in enumerate(filtered_nodes[:top_k]):
                try:
                    # Extract metadata
                    metadata = node.node.metadata if hasattr(node.node, 'metadata') else {}
                    filename = metadata.get('file_name', 'Unknown')
                    
                    # Get content
                    content = node.node.text if hasattr(node.node, 'text') else str(node.node)
                    
                    # Get similarity score
                    similarity_score = node.score if hasattr(node, 'score') else 0.0
                    
                    # DIAGNOSTIC: Check if query terms are in content
                    content_lower = content.lower()
                    contains_query = query_lower in content_lower
                    
                    # Extract entity terms for checking
                    if 'breeda' in query_lower and 'daly' in query_lower:
                        contains_breeda = 'breeda' in content_lower
                        contains_daly = 'daly' in content_lower
                        contains_both = contains_breeda and contains_daly
                        
                        logger.info(f"  ?? {i+1}. {filename} (score: {similarity_score:.3f})")
                        logger.info(f"       Contains 'breeda': {contains_breeda}")
                        logger.info(f"       Contains 'daly': {contains_daly}")
                        logger.info(f"       Contains both: {contains_both}")
                        logger.info(f"       Contains full query: {contains_query}")
                        logger.info(f"       Content preview: {content[:100].replace(chr(10), ' ')}...")
                        
                        # CRITICAL: If this result doesn't contain the person, flag it
                        if not contains_both:
                            logger.warning(f"??  PROBLEM: Document '{filename}' has high similarity ({similarity_score:.3f}) but doesn't contain 'Breeda Daly'!")
                            logger.warning(f"      This suggests vector search is returning irrelevant results!")
                    else:
                        logger.info(f"  ?? {i+1}. {filename} (score: {similarity_score:.3f})")
                        logger.info(f"       Contains query: {contains_query}")
                        logger.info(f"       Content preview: {content[:100].replace(chr(10), ' ')}...")
                    
                    result = RetrievalResult(
                        content=content[:500] + "..." if len(content) > 500 else content,
                        full_content=content,
                        filename=filename,
                        similarity_score=similarity_score,
                        metadata=metadata,
                        source_method=self.get_name(),
                        document_id=metadata.get('id', ''),
                        chunk_index=metadata.get('chunk_index', 0)
                    )
                    
                    # Add diagnostic metadata
                    result.metadata.update({
                        "diagnostic_contains_query": contains_query,
                        "diagnostic_query_used": query,
                        "diagnostic_similarity_threshold": similarity_threshold
                    })
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"?? Error processing node {i}: {e}")
                    continue
            
            # Final diagnostic summary
            relevant_results = [r for r in results if r.metadata.get("diagnostic_contains_query", False)]
            logger.info(f"?? DIAGNOSTIC SUMMARY:")
            logger.info(f"   Total candidates: {len(nodes)}")
            logger.info(f"   After similarity filter: {len(filtered_nodes)}")
            logger.info(f"   Final results: {len(results)}")
            logger.info(f"   Actually relevant: {len(relevant_results)}")
            logger.info(f"   Relevance ratio: {len(relevant_results)/len(results)*100 if results else 0:.1f}%")
            
            if len(relevant_results) == 0 and len(results) > 0:
                logger.error(f"?? CRITICAL ISSUE: Vector search returned {len(results)} results but NONE contain the query terms!")
                logger.error(f"    This indicates a problem with:")
                logger.error(f"    1. Embedding model understanding")
                logger.error(f"    2. Similarity threshold too low") 
                logger.error(f"    3. Database indexing issues")
                logger.error(f"    4. Query preprocessing problems")
            
            return results
            
        except Exception as e:
            logger.error(f"? LlamaIndex retrieval failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

class DatabaseRetriever(BaseRetriever):
    """Direct database retriever for fallback with diagnostic logging"""
    
    def __init__(self, config):
        self.config = config
    
    def is_available(self) -> bool:
        """Database retriever is always available"""
        return True
    
    def get_name(self) -> str:
        return "database_direct"
    
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[RetrievalResult]:
        """Retrieve using direct database search with diagnostic logging"""
        logger.info(f"??? DIAGNOSTIC: Database fallback search for '{query}'")
        
        try:
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(self.config.database.connection_string)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Simple text search as fallback
            search_sql = f"""
            SELECT 
                id,
                metadata,
                (metadata->>'text') as text_content,
                (metadata->>'file_name') as file_name,
                (metadata->>'file_path') as file_path
            FROM {self.config.database.schema}.{self.config.database.table_name}
            WHERE LOWER(metadata->>'text') LIKE LOWER(%s)
            ORDER BY LENGTH(metadata->>'text') ASC
            LIMIT %s
            """
            
            # Use wildcards for broader matching
            search_term = f"%{query}%"
            logger.info(f"?? SQL search with term: {search_term}")
            
            cur.execute(search_sql, (search_term, top_k))
            rows = cur.fetchall()
            
            logger.info(f"?? Database found {len(rows)} direct matches")
            
            cur.close()
            conn.close()
            
            # Convert to RetrievalResult objects
            results = []
            for i, row in enumerate(rows):
                try:
                    content = row.get('text_content', '')
                    if not content:
                        continue
                    
                    metadata = row.get('metadata', {})
                    filename = row.get('file_name') or metadata.get('file_name', 'Unknown')
                    
                    # Calculate simple relevance score based on query occurrence
                    relevance_score = self._calculate_relevance(content, query)
                    
                    logger.info(f"  ?? {i+1}. {filename} (relevance: {relevance_score:.3f})")
                    logger.info(f"       Direct match found in database")
                    
                    result = RetrievalResult(
                        content=content[:500] + "..." if len(content) > 500 else content,
                        full_content=content,
                        filename=filename,
                        similarity_score=relevance_score,
                        metadata=metadata,
                        source_method=self.get_name(),
                        document_id=str(row.get('id', '')),
                        chunk_index=metadata.get('chunk_index', 0)
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"?? Error processing database row {i}: {e}")
                    continue
            
            logger.info(f"? Database retrieval completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"? Database retrieval failed: {e}")
            return []
    
    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate simple relevance score"""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Count occurrences
        occurrences = content_lower.count(query_lower)
        if occurrences == 0:
            return 0.1
        
        # Normalize by content length
        relevance = min(1.0, occurrences / (len(content.split()) / 100))
        return relevance

class ContentFilterRetriever(BaseRetriever):
    """Content-filtering retriever that requires specific terms"""
    
    def __init__(self, base_retriever: BaseRetriever, required_terms: List[str]):
        self.base_retriever = base_retriever
        self.required_terms = [term.lower() for term in required_terms]
    
    def is_available(self) -> bool:
        return self.base_retriever.is_available()
    
    def get_name(self) -> str:
        return f"{self.base_retriever.get_name()}_filtered"
    
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[RetrievalResult]:
        """Retrieve and filter by required terms with diagnostic logging"""
        logger.info(f"?? DIAGNOSTIC: Content filtering with required terms: {self.required_terms}")
        
        # Get base results
        base_results = await self.base_retriever.retrieve(query, top_k * 3, **kwargs)  # Get more for filtering
        logger.info(f"?? Base retriever returned {len(base_results)} results")
        
        if not self.required_terms:
            logger.info("?? No required terms specified, returning base results")
            return base_results[:top_k]
        
        # Apply content filtering
        filtered_results = []
        for i, result in enumerate(base_results):
            full_text = f"{result.content} {result.filename} {result.full_content}".lower()
            
            # Check if ANY required term is present (flexible filtering)
            found_terms = []
            for term in self.required_terms:
                if term in full_text:
                    found_terms.append(term)
            
            logger.info(f"  ?? {i+1}. {result.filename}")
            logger.info(f"       Required terms: {self.required_terms}")
            logger.info(f"       Found terms: {found_terms}")
            logger.info(f"       Match ratio: {len(found_terms)}/{len(self.required_terms)}")
            
            if found_terms:  # At least one term found
                # Update metadata with filtering info
                result.metadata.update({
                    "content_filtered": True,
                    "found_terms": found_terms,
                    "match_ratio": len(found_terms) / len(self.required_terms),
                    "required_terms": self.required_terms
                })
                result.source_method = self.get_name()
                filtered_results.append(result)
                logger.info(f"       ? PASSED content filter")
            else:
                logger.info(f"       ? FAILED content filter (no required terms found)")
        
        # Sort by match ratio and similarity
        filtered_results.sort(
            key=lambda x: (x.metadata.get("match_ratio", 0), x.similarity_score),
            reverse=True
        )
        
        logger.info(f"?? Content filtering result: {len(filtered_results)}/{len(base_results)} results passed filter")
        return filtered_results[:top_k]

class MultiStrategyRetriever:
    """Multi-strategy retriever with diagnostic logging and timeout fixes"""
    
    def __init__(self, config):
        self.config = config
        self.retrievers = {}
        self._initialize_retrievers()
    
    def _initialize_retrievers(self):
        """Initialize all available retrievers"""
        # Primary LlamaIndex retriever
        llamaindex_retriever = LlamaIndexRetriever(self.config)
        if llamaindex_retriever.is_available():
            self.retrievers["llamaindex"] = llamaindex_retriever
        
        # Fallback database retriever
        self.retrievers["database"] = DatabaseRetriever(self.config)
        
        logger.info(f"?? Initialized retrievers: {list(self.retrievers.keys())}")
    
    async def multi_retrieve(self, 
                           queries: List[str], 
                           extracted_entity: Optional[str] = None,
                           required_terms: List[str] = None) -> MultiRetrievalResult:
        """Retrieve using multiple strategies with comprehensive diagnostic logging"""
        start_time = time.time()
        all_results = []
        methods_used = []
        
        logger.info(f"?? DIAGNOSTIC: Starting multi-strategy retrieval")
        logger.info(f"   Queries: {queries}")
        logger.info(f"   Extracted entity: {extracted_entity}")
        logger.info(f"   Required terms: {required_terms}")
        
        # Get dynamic search parameters
        search_params = self.config.get_dynamic_search_params(
            queries[0] if queries else "", 
            extracted_entity
        )
        
        top_k = search_params["top_k"]
        similarity_threshold = search_params["similarity_threshold"]
        
        logger.info(f"?? Dynamic search parameters:")
        logger.info(f"   Top K: {top_k}")
        logger.info(f"   Similarity threshold: {similarity_threshold}")
        
        # Strategy 1: Primary retriever with all query variants
        logger.info(f"?? STRATEGY 1: Primary retriever with query variants")
        primary_results = await self._retrieve_with_variants(
            queries, top_k, similarity_threshold
        )
        
        if primary_results:
            all_results.extend(primary_results)
            methods_used.append("multi_query_primary")
            logger.info(f"? Strategy 1 succeeded: {len(primary_results)} results")
        else:
            logger.warning(f"?? Strategy 1 failed: no results from primary retriever")
        
        # Strategy 2: Content-filtered retrieval (if required terms provided)
        if required_terms and "llamaindex" in self.retrievers:
            logger.info(f"?? STRATEGY 2: Content-filtered retrieval")
            filtered_retriever = ContentFilterRetriever(
                self.retrievers["llamaindex"], 
                required_terms
            )
            
            filtered_results = await filtered_retriever.retrieve(
                queries[0], top_k, similarity_threshold=similarity_threshold
            )
            
            if filtered_results:
                all_results.extend(filtered_results)
                methods_used.append("content_filtered")
                logger.info(f"? Strategy 2 succeeded: {len(filtered_results)} filtered results")
            else:
                logger.warning(f"?? Strategy 2 failed: no results after content filtering")
        
        # Strategy 3: Fallback database search (if primary failed)
        if not all_results and "database" in self.retrievers:
            logger.info(f"??? STRATEGY 3: Fallback database search")
            fallback_results = await self.retrievers["database"].retrieve(
                queries[0], top_k
            )
            
            if fallback_results:
                all_results.extend(fallback_results)
                methods_used.append("database_fallback")
                logger.info(f"? Strategy 3 succeeded: {len(fallback_results)} database results")
            else:
                logger.warning(f"?? Strategy 3 failed: no database results")
        
        # Fuse and rank results
        logger.info(f"?? FUSION: Processing {len(all_results)} total results")
        final_results = self._fuse_results(all_results, queries[0])
        
        retrieval_time = time.time() - start_time
        
        # Final diagnostic summary
        logger.info(f"?? MULTI-RETRIEVAL COMPLETED:")
        logger.info(f"   Total candidates found: {len(all_results)}")
        logger.info(f"   Final results after fusion: {len(final_results)}")
        logger.info(f"   Methods used: {', '.join(methods_used)}")
        logger.info(f"   Total time: {retrieval_time:.3f}s")
        logger.info(f"   Success rate: {len(final_results)/len(all_results)*100 if all_results else 0:.1f}%")
        
        if len(final_results) == 0:
            logger.error(f"?? CRITICAL: No results returned from any strategy!")
            logger.error(f"   Possible issues:")
            logger.error(f"   1. Query terms not in database")
            logger.error(f"   2. Similarity thresholds too high")
            logger.error(f"   3. Embedding model problems")
            logger.error(f"   4. Database connectivity issues")
        
        return MultiRetrievalResult(
            query=queries[0],
            results=final_results,
            methods_used=methods_used,
            total_candidates=len(all_results),
            retrieval_time=retrieval_time,
            fusion_method=self.config.search.fusion_method,
            metadata={
                "search_params": search_params,
                "queries_used": queries,
                "extracted_entity": extracted_entity,
                "required_terms": required_terms,
                "diagnostic_mode": True
            }
        )
    
    async def _retrieve_with_variants(self, 
                                    queries: List[str], 
                                    top_k: int, 
                                    similarity_threshold: float) -> List[RetrievalResult]:
        """Retrieve using multiple query variants with timeout protection"""
        if "llamaindex" not in self.retrievers:
            logger.warning("? LlamaIndex retriever not available for query variants")
            return []
        
        retriever = self.retrievers["llamaindex"]
        all_results = []
        
        logger.info(f"?? Retrieving with {len(queries)} query variants")
        
        # ??????????: ???????? timeout protection ??? ???????????? ????????
        if self.config.search.enable_multi_retrieval:
            logger.info("? Using parallel retrieval")
            tasks = []
            for i, query in enumerate(queries[:self.config.search.max_query_variants]):
                logger.info(f"   ?? Variant {i+1}: '{query}'")
                task = asyncio.wait_for(
                    retriever.retrieve(
                        query, 
                        top_k // len(queries) + 5,  # Distribute top_k among queries
                        similarity_threshold=similarity_threshold
                    ),
                    timeout=30.0  # ??????????: ???????? timeout 30 ??????
                )
                tasks.append(task)
            
            # Execute in parallel with timeout protection
            try:
                results_list = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, results in enumerate(results_list):
                    if isinstance(results, list):
                        logger.info(f"   ? Variant {i+1} returned {len(results)} results")
                        all_results.extend(results)
                    elif isinstance(results, asyncio.TimeoutError):
                        logger.warning(f"   ? Variant {i+1} timed out")
                    else:
                        logger.warning(f"   ? Variant {i+1} failed: {results}")
                        
            except Exception as e:
                logger.error(f"? Parallel retrieval failed: {e}")
                # Fallback to sequential
                logger.info("?? Falling back to sequential retrieval")
                all_results = await self._sequential_retrieval(queries, top_k, similarity_threshold, retriever)
        
        else:
            # Sequential retrieval with timeout protection
            logger.info("?? Using sequential retrieval")
            all_results = await self._sequential_retrieval(queries, top_k, similarity_threshold, retriever)
        
        logger.info(f"?? Query variants summary: {len(all_results)} total results from {len(queries)} variants")
        return all_results
    
    async def _sequential_retrieval(self, queries, top_k, similarity_threshold, retriever):
        """Sequential retrieval with individual timeouts"""
        all_results = []
        
        for i, query in enumerate(queries[:self.config.search.max_query_variants]):
            logger.info(f"   ?? Sequential variant {i+1}: '{query}'")
            try:
                # ??????????: individual timeout ??? ??????? ???????
                results = await asyncio.wait_for(
                    retriever.retrieve(
                        query, 
                        top_k,
                        similarity_threshold=similarity_threshold
                    ),
                    timeout=30.0  # 30 ?????? ?? ?????? ??????
                )
                
                logger.info(f"   ? Variant {i+1} returned {len(results)} results")
                all_results.extend(results)
                
            except asyncio.TimeoutError:
                logger.warning(f"   ? Variant {i+1} timed out after 30s")
            except Exception as e:
                logger.warning(f"   ? Variant {i+1} failed: {e}")
        
        return all_results
    
    def _fuse_results(self, all_results: List[RetrievalResult], original_query: str) -> List[RetrievalResult]:
        """Fuse and rank results from multiple strategies"""
        if not all_results:
            return []
        
        # Remove duplicates based on content hash
        unique_results = {}
        for result in all_results:
            # Create hash based on content and filename
            content_hash = hash(result.full_content + result.filename)
            
            if content_hash not in unique_results:
                unique_results[content_hash] = result
            else:
                # If duplicate, keep the one with higher similarity or better method
                existing = unique_results[content_hash]
                if (result.similarity_score > existing.similarity_score or
                    self._is_better_method(result.source_method, existing.source_method)):
                    unique_results[content_hash] = result
        
        # Apply fusion strategy
        fused_results = list(unique_results.values())
        
        if self.config.search.fusion_method == "weighted_score":
            fused_results = self._weighted_score_fusion(fused_results)
        elif self.config.search.fusion_method == "rank_fusion":
            fused_results = self._rank_fusion(fused_results)
        else:
            # Default: sort by similarity score
            fused_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Limit final results
        return fused_results[:self.config.search.max_final_results]
    
    def _weighted_score_fusion(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Weighted score fusion considering method quality and other factors"""
        method_weights = {
            "llamaindex_vector": 1.0,
            "llamaindex_vector_filtered": 1.2,  # Bonus for content filtering
            "database_direct": 0.7,
            "database_direct_filtered": 0.9
        }
        
        for result in results:
            base_weight = method_weights.get(result.source_method, 1.0)
            
            # Additional weighting factors
            content_boost = 1.0 + result.metadata.get("match_ratio", 0) * 0.3
            
            # Calculate final weighted score
            result.metadata["weighted_score"] = result.similarity_score * base_weight * content_boost
        
        # Sort by weighted score
        results.sort(key=lambda x: x.metadata.get("weighted_score", x.similarity_score), reverse=True)
        return results
    
    def _rank_fusion(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rank-based fusion (RRF - Reciprocal Rank Fusion)"""
        # Group results by method
        method_results = {}
        for result in results:
            method = result.source_method
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(result)
        
        # Sort each method's results
        for method in method_results:
            method_results[method].sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Calculate RRF scores
        rrf_scores = {}
        k = 60  # RRF constant
        
        for result in results:
            content_id = hash(result.full_content + result.filename)
            if content_id not in rrf_scores:
                rrf_scores[content_id] = {"result": result, "score": 0}
            
            # Find rank in its method
            method_list = method_results[result.source_method]
            try:
                rank = method_list.index(result) + 1
                rrf_scores[content_id]["score"] += 1 / (k + rank)
            except ValueError:
                pass
        
        # Sort by RRF score
        fused_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return [item["result"] for item in fused_results]
    
    def _is_better_method(self, method1: str, method2: str) -> bool:
        """Compare which method is better"""
        method_priority = {
            "llamaindex_vector_filtered": 4,
            "llamaindex_vector": 3,
            "database_direct_filtered": 2,
            "database_direct": 1
        }
        
        return method_priority.get(method1, 0) > method_priority.get(method2, 0)
    
    def get_retriever_status(self) -> Dict[str, bool]:
        """Get status of all retrievers"""
        return {name: retriever.is_available() 
                for name, retriever in self.retrievers.items()}