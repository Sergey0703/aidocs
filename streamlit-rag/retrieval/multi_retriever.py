# retrieval/multi_retriever.py
# Multi-strategy retrieval system with UNIVERSAL PERSON NAME DETECTION

import time
import logging
import asyncio
import re
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

class PersonNameDetector:
    """Universal person name detection using best practices from NLP literature"""
    
    def __init__(self):
        # Universal person name patterns from NLP best practices
        self.person_patterns = [
            # Basic: First Last (most common)
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            
            # With middle initial: First M. Last
            r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b',
            
            # With middle name: First Middle Last
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            
            # Hyphenated names: Smith-Jones, Lloyd-Atkinson
            r'\b[A-Z][a-z]+-[A-Z][a-z]+\b',
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+-[A-Z][a-z]+\b',
            
            # Names with apostrophes: D'Angelo, O'Brien
            r"\b[A-Z]'[A-Z][a-z]+\b",
            r"\b[A-Z][a-z]+\s+[A-Z]'[A-Z][a-z]+\b",
            
            # Names with prefixes: Van der, De, Di, etc.
            r'\b[A-Z][a-z]+\s+(?:van|de|di|du|da|del|della|von|zu)\s+[A-Z][a-z]+\b',
            r'\b(?:Van|De|Di|Du|Da|Del|Della|Von|Zu)\s+[A-Z][a-z]+\b',
            
            # Names with suffixes: Jr., Sr., III
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Jr|Sr|III|II|IV)\b',
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.person_patterns]
        
        # Keywords that often indicate person queries
        self.person_keywords = [
            'who is', 'tell me about', 'find', 'about', 'information about',
            'show me', 'give me', 'details about', 'biography', 'profile'
        ]
    
    def is_person_query(self, query: str, extracted_entity: Optional[str] = None) -> bool:
        """Detect if query is about a person using universal patterns"""
        
        # Check extracted entity first (if available)
        if extracted_entity:
            if self.contains_person_name(extracted_entity):
                return True
        
        # Check original query
        if self.contains_person_name(query):
            return True
        
        # Check for person-related keywords + capitalized words
        query_lower = query.lower()
        has_person_keywords = any(keyword in query_lower for keyword in self.person_keywords)
        has_capitalized_words = bool(re.search(r'\b[A-Z][a-z]+\b', query))
        
        return has_person_keywords and has_capitalized_words
    
    def contains_person_name(self, text: str) -> bool:
        """Check if text contains a person name using universal patterns"""
        if not text or len(text.strip()) < 2:
            return False
        
        # Try each compiled pattern
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return True
        
        return False
    
    def extract_person_names(self, text: str) -> List[str]:
        """Extract all person names from text using universal patterns"""
        names = []
        
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            names.extend(matches)
        
        # Remove duplicates while preserving order
        unique_names = []
        for name in names:
            if name not in unique_names:
                unique_names.append(name)
        
        return unique_names
    
    def get_person_name_terms(self, text: str) -> List[str]:
        """Get individual terms from detected person names for content validation"""
        # 1. ??????? ????? ??? person names ????????? regex ????????
        person_names = self.extract_person_names(text)
        
        if not person_names:
            return []  # ???? ???? ??? - ?????????? ?????? ??????
        
        # 2. ????? ??????? ??????? ?????? ?? ????????? ????
        terms = []
        for name in person_names:
            # Split name into individual terms and clean them
            name_terms = [
                term.strip().lower() 
                for term in re.split(r'[\s\-\']', name) 
                if len(term.strip()) > 1
            ]
            terms.extend(name_terms)
        
        # Remove duplicates while preserving order
        unique_terms = []
        for term in terms:
            if term not in unique_terms:
                unique_terms.append(term)
        
        return unique_terms

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
    """LlamaIndex-based vector retriever with UNIVERSAL PERSON NAME SUPPORT"""
    
    def __init__(self, config):
        self.config = config
        self.index = None
        self.embed_model = None
        self.person_detector = PersonNameDetector()
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
    
    def _get_universal_threshold(self, query: str) -> float:
        """Get smart threshold based on UNIVERSAL person name detection"""
        
        # Use universal person detector
        is_person = self.person_detector.is_person_query(query)
        
        if is_person:
            # High threshold for any person names (universal)
            return 0.65
        
        # Check query complexity for non-person queries
        word_count = len(query.split())
        if word_count <= 2:
            return 0.4  # Simple queries
        elif word_count >= 6:
            return 0.3  # Complex queries need lower threshold
        else:
            return 0.35  # Standard queries
    
    async def retrieve(self, query: str, top_k: int = 10, similarity_threshold: float = None, **kwargs) -> List[RetrievalResult]:
        """Retrieve using LlamaIndex with UNIVERSAL person name detection"""
        if not self.is_available():
            logger.warning("? LlamaIndex retriever not available")
            return []
        
        # Use universal threshold if not provided
        if similarity_threshold is None:
            similarity_threshold = self._get_universal_threshold(query)
        
        logger.info(f"?? LlamaIndex retrieval: '{query}' (universal threshold: {similarity_threshold}, top_k: {top_k})")
        
        try:
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core.postprocessor import SimilarityPostprocessor
            
            # Adjust candidates based on threshold
            candidates_multiplier = 3 if similarity_threshold > 0.6 else 2
            candidate_count = top_k * candidates_multiplier
            
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=candidate_count,
                embed_model=self.embed_model
            )
            
            # Create similarity postprocessor
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
            
            # UNIVERSAL content validation
            validated_nodes = []
            
            for node in filtered_nodes:
                try:
                    content = node.node.text if hasattr(node.node, 'text') else str(node.node)
                    
                    # Universal content relevance check
                    if self._is_content_universally_relevant(query, content):
                        validated_nodes.append(node)
                    else:
                        similarity_score = node.score if hasattr(node, 'score') else 0.0
                        metadata = node.node.metadata if hasattr(node.node, 'metadata') else {}
                        filename = metadata.get('file_name', 'Unknown')
                        logger.warning(f"?? FILTERED OUT: {filename} (score: {similarity_score:.3f}) - not universally relevant")
                        
                except Exception as e:
                    logger.warning(f"Error validating node: {e}")
                    continue
            
            logger.info(f"? After universal validation: {len(validated_nodes)} relevant results")
            
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
                    
                    # Add universal validation metadata
                    result.metadata.update({
                        "content_validated": True,
                        "universal_threshold_used": similarity_threshold,
                        "universal_person_detected": self.person_detector.is_person_query(query),
                        "query_validated": query
                    })
                    
                    results.append(result)
                    logger.info(f"  ? {i+1}. {filename} (score: {similarity_score:.3f}) - VALIDATED")
                    
                except Exception as e:
                    logger.warning(f"Error processing validated node {i}: {e}")
                    continue
            
            logger.info(f"?? LlamaIndex final results: {len(results)} universally validated documents")
            return results
            
        except Exception as e:
            logger.error(f"? LlamaIndex retrieval failed: {e}")
            return []
    
    def _is_content_universally_relevant(self, query: str, content: str) -> bool:
        """UNIVERSAL content relevance check using person name detector"""
        
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Check if query contains person names using universal detector
        if self.person_detector.is_person_query(query):
            # ??????????: ?????????? ???????????? ????? get_person_name_terms
            person_terms = self.person_detector.get_person_name_terms(query)
            
            if person_terms:
                # Require ALL person name terms to be in content (strict for person queries)
                found_terms = sum(1 for term in person_terms if term in content_lower)
                return found_terms == len(person_terms)
        
        # For non-person queries - general relevance check
        query_words = [word for word in query_lower.split() if len(word) > 2]
        if not query_words:
            return True
        
        # Require at least 70% of significant words to be in content
        found_words = sum(1 for word in query_words if word in content_lower)
        return found_words / len(query_words) >= 0.7

class DatabaseRetriever(BaseRetriever):
    """Enhanced database retriever with UNIVERSAL person name support"""
    
    def __init__(self, config):
        self.config = config
        self.person_detector = PersonNameDetector()
    
    def is_available(self) -> bool:
        """Database retriever is always available"""
        return True
    
    def get_name(self) -> str:
        return "database_exact"
    
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[RetrievalResult]:
        """Database search with UNIVERSAL person name detection"""
        logger.info(f"??? Database exact search for: '{query}'")
        
        try:
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(self.config.database.connection_string)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            results = []
            
            # Strategy 1: Exact phrase match (highest priority)
            exact_results = await self._exact_phrase_search(cur, query, top_k)
            results.extend(exact_results)
            logger.info(f"?? Exact phrase matches: {len(exact_results)}")
            
            # Strategy 2: Person name terms match (if query contains person names)
            if self.person_detector.is_person_query(query) and len(results) < top_k:
                needed = top_k - len(results)
                person_results = await self._person_name_search(cur, query, needed, exclude_ids=[r.document_id for r in results])
                results.extend(person_results)
                logger.info(f"?? Person name matches: {len(person_results)}")
            
            # Strategy 3: Individual terms match (if still not enough)
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
                
                result = RetrievalResult(
                    content=content[:500] + "..." if len(content) > 500 else content,
                    full_content=content,
                    filename=filename,
                    similarity_score=0.95,  # High score for exact matches
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
    
    async def _person_name_search(self, cur, query: str, limit: int, exclude_ids: List[str] = None) -> List[RetrievalResult]:
        """UNIVERSAL person name based search"""
        if exclude_ids is None:
            exclude_ids = []
        
        # ??????????: ?????????? ???????????? ????? get_person_name_terms
        person_terms = self.person_detector.get_person_name_terms(query)
        if not person_terms:
            return []
        
        logger.info(f"?? Searching for person terms: {person_terms}")
        
        # Build SQL for person name terms
        conditions = []
        params = []
        
        for term in person_terms:
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
                
                # Calculate relevance based on person term coverage
                relevance_score = self._calculate_person_relevance(content, person_terms)
                
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
                    "match_type": "person_name_match",
                    "person_terms": person_terms,
                    "terms_coverage": relevance_score
                })
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing person name match: {e}")
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
    
    def _calculate_person_relevance(self, content: str, person_terms: List[str]) -> float:
        """Calculate relevance for person name matches"""
        content_lower = content.lower()
        found_terms = sum(1 for term in person_terms if term in content_lower)
        
        if found_terms == 0:
            return 0.1
        
        # High score for person matches (requires ALL terms)
        if found_terms == len(person_terms):
            return 0.9
        
        # Partial matches get lower scores
        coverage_score = found_terms / len(person_terms)
        return 0.4 + coverage_score * 0.4
    
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
        
        return min(0.8, 0.4 + coverage_score * 0.3 + occurrence_boost)

class MultiStrategyRetriever:
    """Multi-strategy retriever with UNIVERSAL person name detection"""
    
    def __init__(self, config):
        self.config = config
        self.retrievers = {}
        self.person_detector = PersonNameDetector()
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
        """Multi-strategy retrieval with UNIVERSAL person name detection"""
        start_time = time.time()
        all_results = []
        methods_used = []
        
        primary_query = queries[0] if queries else ""
        
        logger.info(f"?? Universal multi-strategy retrieval")
        logger.info(f"   Primary query: '{primary_query}'")
        logger.info(f"   Entity: '{extracted_entity}'")
        logger.info(f"   Required terms: {required_terms}")
        
        # UNIVERSAL person query detection
        is_person_query = self.person_detector.is_person_query(primary_query, extracted_entity)
        
        # Get universal search parameters
        search_params = self._get_universal_search_params(primary_query, extracted_entity, is_person_query)
        logger.info(f"?? Universal parameters: {search_params}")
        
        # STRATEGY 1: Database exact search (highest priority for person queries)
        if is_person_query and "database" in self.retrievers:
            logger.info(f"??? STRATEGY 1: Database exact search (universal person detected)")
            # ??????????: ?????????? extracted_entity ??? person queries ?????? ??????? query
            search_query = extracted_entity or primary_query
            logger.info(f"   Using search query: '{search_query}' (extracted from: '{primary_query}')")
            database_results = await self.retrievers["database"].retrieve(
                search_query, search_params["top_k"]
            )
            
            if database_results:
                all_results.extend(database_results)
                methods_used.append("database_exact_priority")
                logger.info(f"? Strategy 1: {len(database_results)} exact matches found")
                
                # Early return if sufficient exact matches found
                if len(database_results) >= 3:
                    logger.info("?? Sufficient exact matches found, skipping vector search")
                    final_results = database_results[:search_params["top_k"]]
                    
                    return MultiRetrievalResult(
                        query=primary_query,
                        results=final_results,
                        methods_used=methods_used,
                        total_candidates=len(all_results),
                        retrieval_time=time.time() - start_time,
                        fusion_method="database_priority",
                        metadata={
                            "search_params": search_params,
                            "strategy": "database_priority_success",
                            "universal_person_detected": True
                        }
                    )
        
        # STRATEGY 2: Vector search with universal thresholds
        if "llamaindex" in self.retrievers:
            logger.info(f"?? STRATEGY 2: Vector search with universal threshold")
            
            if is_person_query:
                # For person queries - single precise query with high threshold
                vector_results = await self.retrievers["llamaindex"].retrieve(
                    extracted_entity or primary_query, 
                    search_params["top_k"],
                    similarity_threshold=search_params["similarity_threshold"]
                )
                logger.info(f"   Single person query: '{extracted_entity or primary_query}'")
            else:
                # For general queries - can use multiple variants
                vector_results = await self._retrieve_with_variants(
                    queries[:2], search_params["top_k"], search_params["similarity_threshold"]
                )
                logger.info(f"   Multiple variants for general query")
            
            if vector_results:
                all_results.extend(vector_results)
                methods_used.append("vector_universal_threshold")
                logger.info(f"? Strategy 2: {len(vector_results)} vector results")
        
        # STRATEGY 3: Database fallback (if primary strategies failed)
        if not all_results and "database" in self.retrievers:
            logger.info(f"?? STRATEGY 3: Database fallback")
            # ??????????: ?????????? extracted_entity ? ??? fallback
            search_query = extracted_entity or primary_query
            logger.info(f"   Using fallback search query: '{search_query}'")
            fallback_results = await self.retrievers["database"].retrieve(
                search_query, search_params["top_k"]
            )
            
            if fallback_results:
                all_results.extend(fallback_results)
                methods_used.append("database_fallback")
                logger.info(f"? Strategy 3: {len(fallback_results)} fallback results")
        
        # Simple deduplication and ranking
        final_results = self._universal_dedupe_and_rank(all_results, search_params["top_k"])
        
        retrieval_time = time.time() - start_time
        
        logger.info(f"?? UNIVERSAL RETRIEVAL COMPLETED:")
        logger.info(f"   Query type: {'PERSON' if is_person_query else 'GENERAL'}")
        logger.info(f"   Total candidates: {len(all_results)}")
        logger.info(f"   Final results: {len(final_results)}")
        logger.info(f"   Methods used: {', '.join(methods_used)}")
        logger.info(f"   Time: {retrieval_time:.3f}s")
        
        return MultiRetrievalResult(
            query=primary_query,
            results=final_results,
            methods_used=methods_used,
            total_candidates=len(all_results),
            retrieval_time=retrieval_time,
            fusion_method="universal_multi_strategy",
            metadata={
                "search_params": search_params,
                "universal_person_detected": is_person_query,
                "person_names_found": self.person_detector.extract_person_names(primary_query),
                "strategy": "universal_multi_strategy"
            }
        )
    
    def _get_universal_search_params(self, query: str, extracted_entity: Optional[str], is_person_query: bool) -> Dict:
        """Get universal search parameters based on query analysis"""
        
        if is_person_query:
            # Conservative parameters for person queries
            return {
                "similarity_threshold": 0.65,  # High threshold for person names
                "top_k": 10,
                "strategy": "universal_person_focused"
            }
        else:
            # Adaptive parameters for general queries
            word_count = len(query.split())
            if word_count <= 2:
                return {
                    "similarity_threshold": 0.4,
                    "top_k": 12,
                    "strategy": "universal_simple"
                }
            elif word_count >= 6:
                return {
                    "similarity_threshold": 0.3,
                    "top_k": 18,
                    "strategy": "universal_complex"
                }
            else:
                return {
                    "similarity_threshold": 0.35,
                    "top_k": 15,
                    "strategy": "universal_standard"
                }
    
    async def _retrieve_with_variants(self, 
                                    queries: List[str], 
                                    top_k: int, 
                                    similarity_threshold: float) -> List[RetrievalResult]:
        """Retrieve with query variants (only for non-person queries)"""
        
        if "llamaindex" not in self.retrievers:
            return []
        
        retriever = self.retrievers["llamaindex"]
        all_results = []
        
        # Simple sequential processing for variants
        for i, query in enumerate(queries[:2]):  # Max 2 variants
            try:
                logger.info(f"   ?? Variant {i+1}: '{query}'")
                
                results = await retriever.retrieve(
                    query, 
                    top_k // len(queries) + 2,  # Distribute top_k
                    similarity_threshold=similarity_threshold
                )
                
                if results:
                    all_results.extend(results)
                    logger.info(f"   ? Variant {i+1} returned {len(results)} results")
                else:
                    logger.info(f"   ? Variant {i+1} returned no results")
                    
            except Exception as e:
                logger.warning(f"   ? Variant {i+1} failed: {e}")
                continue
        
        logger.info(f"?? Query variants summary: {len(all_results)} total results")
        return all_results
    
    def _universal_dedupe_and_rank(self, all_results: List[RetrievalResult], max_results: int) -> List[RetrievalResult]:
        """Universal deduplication and ranking"""
        
        if not all_results:
            return []
        
        # Simple deduplication by content hash
        unique_results = {}
        
        for result in all_results:
            # Create hash from content + filename
            content_hash = hash(result.full_content[:200] + result.filename)
            
            if content_hash not in unique_results:
                unique_results[content_hash] = result
            else:
                # Keep the better result
                existing = unique_results[content_hash]
                if (result.similarity_score > existing.similarity_score or
                    self._is_better_method(result.source_method, existing.source_method)):
                    unique_results[content_hash] = result
        
        # Simple ranking with method weights
        final_results = list(unique_results.values())
        
        method_weights = {
            "database_exact": 1.0,
            "llamaindex_vector": 0.9,
        }
        
        for result in final_results:
            weight = method_weights.get(result.source_method, 0.8)
            result.metadata["final_score"] = result.similarity_score * weight
        
        # Sort by final score
        final_results.sort(key=lambda x: x.metadata.get("final_score", x.similarity_score), reverse=True)
        
        logger.info(f"?? Universal deduplication: {len(all_results)} ? {len(final_results)} unique ? {min(len(final_results), max_results)} final")
        
        return final_results[:max_results]
    
    def _is_better_method(self, method1: str, method2: str) -> bool:
        """Compare which method is better"""
        priority = {
            "database_exact": 3,
            "llamaindex_vector": 2,
            "database_fallback": 1
        }
        return priority.get(method1, 0) > priority.get(method2, 0)
    
    def get_retriever_status(self) -> Dict[str, bool]:
        """Get status of all retrievers"""
        return {name: retriever.is_available() 
                for name, retriever in self.retrievers.items()}