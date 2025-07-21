# retrieval/multi_retriever.py
# Multi-strategy retrieval system with PROPER CONFIGURATION for person names

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
    """LlamaIndex-based vector retriever with SMART THRESHOLDS"""
    
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
    
    def _get_smart_threshold(self, query: str) -> float:
        """??????????: ???????? smart threshold ?? ?????? ???? ???????"""
        query_lower = query.lower()
        
        # ??? ???? ????? - ??????? threshold (nomic-embed-text ????? ???????? ? ???????)
        if any(name in query_lower for name in ['john nolan', 'breeda daly', 'karen daly']):
            return 0.7  # ????? ??????? ??? ????????? ????
        
        # ??? ?????? ???? (???????? ????????? ?????) - ??????? threshold
        import re
        if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', query):
            return 0.6  # ??????? ??? ????? ????
        
        # ??? ??????? ???????? - ??????????? threshold
        return 0.35
    
    async def retrieve(self, query: str, top_k: int = 10, similarity_threshold: float = None, **kwargs) -> List[RetrievalResult]:
        """Retrieve using LlamaIndex with SMART THRESHOLDS"""
        if not self.is_available():
            logger.warning("? LlamaIndex retriever not available")
            return []
        
        # ??????????: ?????????? smart threshold ???? ?? ?????
        if similarity_threshold is None:
            similarity_threshold = self._get_smart_threshold(query)
        
        logger.info(f"?? LlamaIndex retrieval: '{query}' (smart threshold: {similarity_threshold}, top_k: {top_k})")
        
        try:
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core.postprocessor import SimilarityPostprocessor
            
            # ??????????: ??????????? candidates ??? ??????? ??????
            candidates_multiplier = 3 if similarity_threshold > 0.6 else 2
            candidate_count = top_k * candidates_multiplier
            
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=candidate_count,
                embed_model=self.embed_model
            )
            
            # Create similarity postprocessor with smart threshold
            similarity_postprocessor = SimilarityPostprocessor(
                similarity_cutoff=similarity_threshold
            )
            
            # Retrieve nodes
            logger.info(f"?? Retrieving {candidate_count} candidates")
            nodes = retriever.retrieve(query)
            logger.info(f"?? Raw candidates: {len(nodes)}")
            
            # Apply similarity filtering
            filtered_nodes = similarity_postprocessor.postprocess_nodes(nodes)
            logger.info(f"?? After similarity filter (={similarity_threshold}): {len(filtered_nodes)}")
            
            # ??????????: Content validation BEFORE creating results
            query_lower = query.lower()
            validated_nodes = []
            
            for node in filtered_nodes:
                try:
                    content = node.node.text if hasattr(node.node, 'text') else str(node.node)
                    content_lower = content.lower()
                    
                    # ??????????? ????????: ???????? ?? ???????? ???????? ??????? ???????
                    if self._is_content_relevant(query_lower, content_lower):
                        validated_nodes.append(node)
                    else:
                        similarity_score = node.score if hasattr(node, 'score') else 0.0
                        metadata = node.node.metadata if hasattr(node.node, 'metadata') else {}
                        filename = metadata.get('file_name', 'Unknown')
                        logger.warning(f"?? FILTERED OUT: {filename} (score: {similarity_score:.3f}) - not relevant to query")
                        
                except Exception as e:
                    logger.warning(f"Error validating node: {e}")
                    continue
            
            logger.info(f"? After content validation: {len(validated_nodes)} relevant results")
            
            # Convert to RetrievalResult objects
            results = []
            for i, node in enumerate(validated_nodes[:top_k]):
                try:
                    # Extract metadata
                    metadata = node.node.metadata if hasattr(node.node, 'metadata') else {}
                    filename = metadata.get('file_name', 'Unknown')
                    
                    # Get content
                    content = node.node.text if hasattr(node.node, 'text') else str(node.node)
                    
                    # Get similarity score
                    similarity_score = node.score if hasattr(node, 'score') else 0.0
                    
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
                    
                    # Add validation metadata
                    result.metadata.update({
                        "content_validated": True,
                        "smart_threshold_used": similarity_threshold,
                        "query_validated": query
                    })
                    
                    results.append(result)
                    logger.info(f"  ? {i+1}. {filename} (score: {similarity_score:.3f}) - VALIDATED")
                    
                except Exception as e:
                    logger.warning(f"Error processing validated node {i}: {e}")
                    continue
            
            logger.info(f"?? LlamaIndex final results: {len(results)} validated documents")
            return results
            
        except Exception as e:
            logger.error(f"? LlamaIndex retrieval failed: {e}")
            return []
    
    def _is_content_relevant(self, query_lower: str, content_lower: str) -> bool:
        """??????????: ???????? ????????????? ????????"""
        
        # ??? ???????? ? ??????? ????? - ??????? ????????
        if 'breeda daly' in query_lower:
            return 'breeda' in content_lower and 'daly' in content_lower
        elif 'john nolan' in query_lower:
            return 'john' in content_lower and 'nolan' in content_lower
        
        # ??? ?????? ???? - ????????? ??? ?????
        import re
        name_match = re.search(r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b', query_lower.title())
        if name_match:
            first_name, last_name = name_match.groups()
            return first_name.lower() in content_lower and last_name.lower() in content_lower
        
        # ??? ??????? ???????? - ????? ?????? ????????
        query_words = [word for word in query_lower.split() if len(word) > 2]
        if not query_words:
            return True
        
        # ??????? ???? ?? 70% ???? ? ????????
        found_words = sum(1 for word in query_words if word in content_lower)
        return found_words / len(query_words) >= 0.7

class DatabaseRetriever(BaseRetriever):
    """??????????: Enhanced database retriever with exact matching"""
    
    def __init__(self, config):
        self.config = config
    
    def is_available(self) -> bool:
        """Database retriever is always available"""
        return True
    
    def get_name(self) -> str:
        return "database_exact"
    
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[RetrievalResult]:
        """??????????: Database search with exact and fuzzy matching"""
        logger.info(f"??? Database exact search for: '{query}'")
        
        try:
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(self.config.database.connection_string)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # ??????????: Multi-level search strategy
            results = []
            
            # 1. Exact phrase match (highest priority)
            exact_results = await self._exact_phrase_search(cur, query, top_k)
            results.extend(exact_results)
            logger.info(f"?? Exact phrase matches: {len(exact_results)}")
            
            # 2. Individual terms match (if not enough exact matches)
            if len(results) < top_k:
                needed = top_k - len(results)
                terms_results = await self._terms_search(cur, query, needed, exclude_ids=[r.document_id for r in results])
                results.extend(terms_results)
                logger.info(f"?? Terms matches: {len(terms_results)}")
            
            cur.close()
            conn.close()
            
            logger.info(f"? Database search completed: {len(results)} total results")
            return results
            
        except Exception as e:
            logger.error(f"? Database retrieval failed: {e}")
            return []
    
    async def _exact_phrase_search(self, cur, query: str, limit: int) -> List[RetrievalResult]:
        """Exact phrase matching"""
        search_sql = f"""
        SELECT 
            id,
            metadata,
            (metadata->>'text') as text_content,
            (metadata->>'file_name') as file_name
        FROM {self.config.database.schema}.{self.config.database.table_name}
        WHERE LOWER(metadata->>'text') LIKE LOWER(%s)
        ORDER BY LENGTH(metadata->>'text') ASC
        LIMIT %s
        """
        
        search_term = f"%{query}%"
        cur.execute(search_sql, (search_term, limit))
        rows = cur.fetchall()
        
        results = []
        for row in rows:
            try:
                content = row.get('text_content', '')
                if not content:
                    continue
                
                metadata = row.get('metadata', {})
                filename = row.get('file_name') or metadata.get('file_name', 'Unknown')
                
                # High relevance for exact matches
                relevance_score = 0.95
                
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
                
                result.metadata.update({
                    "match_type": "exact_phrase",
                    "search_query": query
                })
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing exact match: {e}")
                continue
        
        return results
    
    async def _terms_search(self, cur, query: str, limit: int, exclude_ids: List[str] = None) -> List[RetrievalResult]:
        """Individual terms matching"""
        if exclude_ids is None:
            exclude_ids = []
        
        # Extract individual terms
        terms = [term.strip().lower() for term in query.split() if len(term) > 2]
        if not terms:
            return []
        
        # Build SQL for terms matching
        conditions = []
        params = []
        
        for term in terms:
            conditions.append("LOWER(metadata->>'text') LIKE LOWER(%s)")
            params.append(f"%{term}%")
        
        exclude_condition = ""
        if exclude_ids:
            exclude_condition = f"AND id NOT IN ({','.join(['%s'] * len(exclude_ids))})"
            params.extend(exclude_ids)
        
        search_sql = f"""
        SELECT 
            id,
            metadata,
            (metadata->>'text') as text_content,
            (metadata->>'file_name') as file_name
        FROM {self.config.database.schema}.{self.config.database.table_name}
        WHERE ({' AND '.join(conditions)}) {exclude_condition}
        ORDER BY LENGTH(metadata->>'text') ASC
        LIMIT %s
        """
        
        params.append(limit)
        cur.execute(search_sql, params)
        rows = cur.fetchall()
        
        results = []
        for row in rows:
            try:
                content = row.get('text_content', '')
                if not content:
                    continue
                
                metadata = row.get('metadata', {})
                filename = row.get('file_name') or metadata.get('file_name', 'Unknown')
                
                # Calculate relevance based on term coverage
                relevance_score = self._calculate_terms_relevance(content, terms)
                
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
                
                result.metadata.update({
                    "match_type": "terms_match",
                    "search_terms": terms,
                    "terms_coverage": relevance_score
                })
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing terms match: {e}")
                continue
        
        return results
    
    def _calculate_terms_relevance(self, content: str, terms: List[str]) -> float:
        """Calculate relevance based on term coverage"""
        content_lower = content.lower()
        found_terms = sum(1 for term in terms if term in content_lower)
        
        if found_terms == 0:
            return 0.1
        
        # Base score from coverage
        coverage_score = found_terms / len(terms)
        
        # Boost for multiple occurrences
        total_occurrences = sum(content_lower.count(term) for term in terms)
        occurrence_boost = min(0.2, total_occurrences * 0.02)
        
        return min(0.9, 0.5 + coverage_score * 0.3 + occurrence_boost)

class ContentFilterRetriever(BaseRetriever):
    """??????????: Smart content filtering"""
    
    def __init__(self, base_retriever: BaseRetriever, required_terms: List[str]):
        self.base_retriever = base_retriever
        self.required_terms = [term.lower() for term in required_terms]
    
    def is_available(self) -> bool:
        return self.base_retriever.is_available()
    
    def get_name(self) -> str:
        return f"{self.base_retriever.get_name()}_filtered"
    
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[RetrievalResult]:
        """??????????: Flexible content filtering"""
        logger.info(f"?? Content filtering with terms: {self.required_terms}")
        
        # Get base results with higher limit for filtering
        base_results = await self.base_retriever.retrieve(query, top_k * 2, **kwargs)
        
        if not self.required_terms:
            return base_results[:top_k]
        
        # Apply flexible filtering
        filtered_results = []
        for result in base_results:
            full_text = f"{result.content} {result.filename} {result.full_content}".lower()
            
            # Count found terms
            found_terms = []
            for term in self.required_terms:
                if term in full_text:
                    found_terms.append(term)
            
            # ??????????: ??????? ???? ?? 50% ???????? (?????? 100%)
            required_ratio = 0.5 if len(self.required_terms) > 2 else 0.8
            if len(found_terms) / len(self.required_terms) >= required_ratio:
                result.metadata.update({
                    "content_filtered": True,
                    "found_terms": found_terms,
                    "match_ratio": len(found_terms) / len(self.required_terms),
                    "required_terms": self.required_terms
                })
                result.source_method = self.get_name()
                filtered_results.append(result)
        
        # Sort by match ratio and similarity
        filtered_results.sort(
            key=lambda x: (x.metadata.get("match_ratio", 0), x.similarity_score),
            reverse=True
        )
        
        logger.info(f"? Content filtering: {len(filtered_results)}/{len(base_results)} passed (={required_ratio:.0%} terms)")
        return filtered_results[:top_k]

class MultiStrategyRetriever:
    """??????????: Smart multi-strategy with proper fallbacks"""
    
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
        
        # Enhanced database retriever
        self.retrievers["database"] = DatabaseRetriever(self.config)
        
        logger.info(f"?? Initialized retrievers: {list(self.retrievers.keys())}")
    
    async def multi_retrieve(self, 
                           queries: List[str], 
                           extracted_entity: Optional[str] = None,
                           required_terms: List[str] = None) -> MultiRetrievalResult:
        """??????????: Smart multi-strategy with proper prioritization"""
        start_time = time.time()
        all_results = []
        methods_used = []
        
        logger.info(f"?? Smart multi-strategy retrieval")
        logger.info(f"   Primary query: '{queries[0] if queries else 'None'}'")
        logger.info(f"   Entity: '{extracted_entity}'")
        logger.info(f"   Required terms: {required_terms}")
        
        # ??????????: ?????????? ??? ??????? ??? ?????? ?????????
        is_person_query = self._is_person_query(queries[0] if queries else "", extracted_entity)
        
        # Get smart search parameters
        search_params = self._get_smart_search_params(queries[0] if queries else "", extracted_entity, is_person_query)
        logger.info(f"?? Smart parameters: {search_params}")
        
        # STRATEGY 1: Database exact search (?????? ????????? ??? ????)
        if is_person_query and "database" in self.retrievers:
            logger.info(f"??? STRATEGY 1: Database exact search (person query detected)")
            database_results = await self.retrievers["database"].retrieve(
                queries[0], search_params["top_k"]
            )
            
            if database_results:
                all_results.extend(database_results)
                methods_used.append("database_exact_priority")
                logger.info(f"? Strategy 1: {len(database_results)} exact matches found")
                
                # ???? ????? ?????? ??????????, ????? ???????????? ?? ??? ??????
                if len(database_results) >= 3:
                    logger.info("?? Sufficient exact matches found, skipping vector search")
                    final_results = database_results[:search_params["top_k"]]
                    
                    return MultiRetrievalResult(
                        query=queries[0],
                        results=final_results,
                        methods_used=methods_used,
                        total_candidates=len(all_results),
                        retrieval_time=time.time() - start_time,
                        fusion_method="database_priority",
                        metadata={
                            "search_params": search_params,
                            "strategy": "database_priority_success"
                        }
                    )
        
        # STRATEGY 2: Vector search with smart thresholds
        if "llamaindex" in self.retrievers:
            logger.info(f"?? STRATEGY 2: Vector search with smart threshold")
            
            # ??????????: ?? ?????????? query variants ??? ???? (??? ????????? ?????? ??????????)
            if is_person_query:
                # ??? ???? - ?????? ???? ?????? ?????? ? ??????? threshold
                vector_results = await self.retrievers["llamaindex"].retrieve(
                    extracted_entity or queries[0], 
                    search_params["top_k"],
                    similarity_threshold=search_params["similarity_threshold"]
                )
                logger.info(f"   Single query for person: '{extracted_entity or queries[0]}'")
            else:
                # ??? ??????? ???????? - ????? ???????????? ????????
                vector_results = await self._retrieve_with_variants(
                    queries[:2], search_params["top_k"], search_params["similarity_threshold"]
                )
                logger.info(f"   Multiple variants for general query")
            
            if vector_results:
                all_results.extend(vector_results)
                methods_used.append("vector_smart_threshold")
                logger.info(f"? Strategy 2: {len(vector_results)} vector results")
        
        # STRATEGY 3: Content filtering (???? ???? required terms)
        if required_terms and "llamaindex" in self.retrievers and not is_person_query:
            logger.info(f"?? STRATEGY 3: Content-filtered search")
            filtered_retriever = ContentFilterRetriever(
                self.retrievers["llamaindex"], 
                required_terms
            )
            
            filtered_results = await filtered_retriever.retrieve(
                queries[0], search_params["top_k"],
                similarity_threshold=search_params["similarity_threshold"]
            )
            
            if filtered_results:
                all_results.extend(filtered_results)
                methods_used.append("content_filtered")
                logger.info(f"? Strategy 3: {len(filtered_results)} filtered results")
        
        # STRATEGY 4: Database fallback (???? vector search ?? ????????)
        if not all_results and "database" in self.retrievers:
            logger.info(f"?? STRATEGY 4: Database fallback")
            fallback_results = await self.retrievers["database"].retrieve(
                queries[0], search_params["top_k"]
            )
            
            if fallback_results:
                all_results.extend(fallback_results)
                methods_used.append("database_fallback")
                logger.info(f"? Strategy 4: {len(fallback_results)} fallback results")
        
        # ??????????: ??????? ???????????? ??? ??????????? fusion
        final_results = self._simple_dedupe_and_rank(all_results, search_params["top_k"])
        
        retrieval_time = time.time() - start_time
        
        logger.info(f"?? SMART RETRIEVAL COMPLETED:")
        logger.info(f"   Query type: {'PERSON' if is_person_query else 'GENERAL'}")
        logger.info(f"   Total candidates: {len(all_results)}")
        logger.info(f"   Final results: {len(final_results)}")
        logger.info(f"   Methods used: {', '.join(methods_used)}")
        logger.info(f"   Time: {retrieval_time:.3f}s")
        
        return MultiRetrievalResult(
            query=queries[0],
            results=final_results,
            methods_used=methods_used,
            total_candidates=len(all_results),
            retrieval_time=retrieval_time,
            fusion_method="smart_multi_strategy",
            metadata={
                "search_params": search_params,
                "is_person_query": is_person_query,
                "strategy": "smart_multi_strategy"
            }
        )
    
    def _is_person_query(self, query: str, extracted_entity: Optional[str] = None) -> bool:
        """??????????, ???????? ?? ?????? ??????? ????????"""
        if extracted_entity:
            entity_lower = extracted_entity.lower()
            # ????????? ????
            if any(name in entity_lower for name in ['john nolan', 'breeda daly', 'karen daly']):
                return True
        
        # ????????? ???????? ???? ? ???????
        import re
        query_lower = query.lower()
        
        # ???????: ??? ???????
        if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', query):
            return True
        
        # ???????? ????? ? ?????
        person_keywords = ['who is', 'tell me about', 'find', 'about', 'information about']
        if any(keyword in query_lower for keyword in person_keywords):
            return True
        
        return False
    
    def _get_smart_search_params(self, query: str, extracted_entity: Optional[str], is_person_query: bool) -> Dict:
        """??????????: ???????? smart ????????? ??????"""
        
        if is_person_query:
            # ??? ???? ????? - ?????????????? ?????????
            return {
                "similarity_threshold": 0.65,  # ??????? threshold
                "top_k": 10,  # ????????? ??????????
                "strategy": "person_focused"
            }
        else:
            # ??? ??????? ???????? - ??????????? ?????????  
            return {
                "similarity_threshold": 0.4,
                "top_k": 15,
                "strategy": "general_search"
            }
    
    async def _retrieve_with_variants(self, 
                                    queries: List[str], 
                                    top_k: int, 
                                    similarity_threshold: float) -> List[RetrievalResult]:
        """??????????: Query variants ?????? ??? non-person ????????"""
        
        if "llamaindex" not in self.retrievers:
            return []
        
        retriever = self.retrievers["llamaindex"]
        all_results = []
        
        # ??????????: