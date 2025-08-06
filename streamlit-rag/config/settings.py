# config/settings.py
# Configuration settings for Production RAG System with Hybrid Search

import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    connection_string: str
    table_name: str = "documents"
    schema: str = "vecs"

@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "nomic-embed-text"
    dimension: int = 768
    base_url: str = "http://localhost:11434"

@dataclass
class LLMConfig:
    """LLM configuration for various purposes"""
    # Main LLM for answer generation
    main_model: str = "llama3.2:3b"
    main_base_url: str = "http://localhost:11434"
    main_timeout: float = 60.0
    
    # Entity extraction LLM (more precise)
    extraction_model: str = "llama3:8b-instruct-q4_K_M"
    extraction_base_url: str = "http://localhost:11434"
    extraction_timeout: float = 30.0
    extraction_temperature: float = 0.0
    extraction_max_tokens: int = 10
    
    # Query rewriting LLM (creative)
    rewrite_model: str = "llama3.2:3b"
    rewrite_base_url: str = "http://localhost:11434"
    rewrite_timeout: float = 20.0
    rewrite_temperature: float = 0.3
    rewrite_max_tokens: int = 100

@dataclass
class SearchConfig:
    """Search and retrieval configuration with Hybrid Search"""
    
    # ?? HYBRID SEARCH SETTINGS
    enable_hybrid_search: bool = True
    enable_vector_search: bool = True
    enable_database_search: bool = True
    
    # Vector search thresholds (lowered for better recall)
    default_similarity_threshold: float = 0.30  # Lowered from 0.35
    entity_similarity_threshold: float = 0.25   # Lowered from 0.30
    fallback_similarity_threshold: float = 0.20 # Lowered from 0.25
    
    # Vector search top_k (respecting 1000 limit)
    default_top_k: int = 20
    entity_top_k: int = 50
    complex_query_top_k: int = 30
    vector_max_top_k: int = 1000  # Supabase/vecs hard limit
    
    # ?? DATABASE SEARCH SETTINGS
    database_search_enabled: bool = True
    database_max_results: int = 100
    database_exact_match_score: float = 0.95  # High score for exact matches
    database_base_score: float = 0.60         # Base score for database results
    database_score_per_occurrence: float = 0.05  # Bonus per query occurrence
    
    # Multi-query settings
    max_query_variants: int = 3
    enable_query_rewriting: bool = True
    enable_entity_extraction: bool = True
    enable_multi_retrieval: bool = True
    
    # Results fusion with hybrid support
    min_results_for_fusion: int = 2
    max_final_results: int = 20  # Increased from 15
    fusion_method: str = "hybrid_weighted"  # Changed from "weighted_score"
    
    # ?? HYBRID FUSION WEIGHTS
    vector_result_weight: float = 0.7
    database_result_weight: float = 1.0      # Database gets higher weight
    exact_match_boost: float = 1.3           # Boost for exact entity matches
    person_name_boost: float = 1.2           # Boost for person name queries
    
    # ?? SEARCH STRATEGY SELECTION
    person_query_strategy: str = "database_priority"  # Prioritize DB for person names
    general_query_strategy: str = "vector_priority"   # Prioritize vector for general queries
    hybrid_merge_strategy: str = "score_weighted"     # How to merge results

@dataclass
class EntityExtractionConfig:
    """Entity extraction configuration"""
    extraction_methods: List[str] = None
    fallback_enabled: bool = True
    validation_enabled: bool = True
    
    # Known entities for special handling (updated for hybrid search)
    known_entities: Dict[str, Dict] = None
    
    # Extraction prompts
    person_extraction_prompt: str = """Extract only the person's name from this question. Return ONLY the name, no other words.

Examples:
- "tell me about John Smith" -> John Smith
- "who is Mary Johnson" -> Mary Johnson  
- "find information about Bob Wilson" -> Bob Wilson
- "show me John Nolan" -> John Nolan
- "John Nolan certifications" -> John Nolan

Question: {query}

Name:"""
    
    def __post_init__(self):
        if self.extraction_methods is None:
            self.extraction_methods = ["llm", "regex", "spacy"]
        
        if self.known_entities is None:
            # Updated with hybrid search parameters
            self.known_entities = {
                "john nolan": {
                    "similarity_threshold": 0.25,  # Lowered for better recall
                    "top_k": 50,
                    "expected_docs": 9,
                    "search_strategy": "hybrid",  # ??
                    "database_priority": True     # ??
                },
                "breeda daly": {
                    "similarity_threshold": 0.25,
                    "top_k": 50,
                    "expected_docs": 20,          # Updated count!
                    "search_strategy": "hybrid",  # ??
                    "database_priority": True     # ??
                },
                "bernie loughnane": {
                    "similarity_threshold": 0.25,
                    "top_k": 50,
                    "expected_docs": 5,
                    "search_strategy": "hybrid",  # ??
                    "database_priority": True     # ??
                }
            }

@dataclass
class QueryRewriteConfig:
    """Query rewriting configuration"""
    enabled: bool = True
    max_rewrites: int = 3
    rewrite_strategies: List[str] = None
    
    # ?? HYBRID SEARCH AWARE REWRITING
    hybrid_rewrite_enabled: bool = True
    entity_query_simplification: bool = True  # Simplify person name queries
    
    # Rewrite prompts
    expand_query_prompt: str = """Generate {num_queries} different ways to search for information about this topic. Make each query more specific and focused.

Original query: {query}

Generate {num_queries} search variations:"""
    
    simplify_query_prompt: str = """Simplify this query to extract the core search terms while preserving the meaning.

Complex query: {query}

Simplified query:"""
    
    # ?? ENTITY-SPECIFIC REWRITING
    person_query_simplification_prompt: str = """This appears to be a query about a person. Extract just the person's name for the most effective search.

Original query: {query}

Person name:"""
    
    def __post_init__(self):
        if self.rewrite_strategies is None:
            self.rewrite_strategies = ["expand", "simplify", "rephrase", "entity_extract"]  # Added entity_extract

@dataclass
class UIConfig:
    """Streamlit UI configuration"""
    page_title: str = "Production RAG System"
    page_icon: str = "??"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    
    # Performance settings
    cache_ttl: int = 300  # 5 minutes
    show_debug_info: bool = True
    show_performance_metrics: bool = True
    show_advanced_settings: bool = True
    
    # ?? HYBRID SEARCH UI SETTINGS
    show_search_strategy_info: bool = True
    show_database_results: bool = True
    show_vector_results: bool = True
    show_hybrid_fusion_details: bool = True
    
    # Example queries (updated with expected counts)
    example_queries: List[str] = None
    
    def __post_init__(self):
        if self.example_queries is None:
            self.example_queries = [
                "John Nolan",
                "tell me about John Nolan",
                "show me John Nolan certifications", 
                "who is Breeda Daly",               # Now finds 20 docs!
                "find Breeda Daly training",        # Now finds 20 docs!
                "what certifications does John Nolan have?",
                "give me information about Breeda Daly's courses",
                "Bernie Loughnane documents"
            ]

class ProductionRAGConfig:
    """Main configuration class for Production RAG System with Hybrid Search"""
    
    def __init__(self):
        # Load environment variables
        self.database = DatabaseConfig(
            connection_string=self._get_connection_string(),
            table_name=os.getenv("TABLE_NAME", "documents")
        )
        
        self.embedding = EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        
        self.llm = LLMConfig(
            main_model=os.getenv("MAIN_LLM_MODEL", "llama3.2:3b"),
            extraction_model=os.getenv("EXTRACTION_LLM_MODEL", "llama3:8b-instruct-q4_K_M"),
            rewrite_model=os.getenv("REWRITE_LLM_MODEL", "llama3.2:3b")
        )
        
        self.search = SearchConfig()
        self.entity_extraction = EntityExtractionConfig()
        self.query_rewrite = QueryRewriteConfig()
        self.ui = UIConfig()
    
    def _get_connection_string(self) -> str:
        """Get database connection string from environment"""
        connection_string = (
            os.getenv("SUPABASE_CONNECTION_STRING") or
            os.getenv("DATABASE_URL") or
            os.getenv("POSTGRES_URL")
        )
        
        if not connection_string:
            raise ValueError("No database connection string found in environment variables!")
        
        return connection_string
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration settings"""
        validation_results = {}
        
        # Check database connection
        validation_results["database_config"] = bool(self.database.connection_string)
        
        # Check embedding configuration
        validation_results["embedding_config"] = bool(
            self.embedding.model_name and 
            self.embedding.base_url and
            self.embedding.dimension > 0
        )
        
        # Check LLM configuration
        validation_results["llm_config"] = bool(
            self.llm.main_model and 
            self.llm.extraction_model and
            self.llm.rewrite_model
        )
        
        # Check search configuration
        validation_results["search_config"] = bool(
            0 < self.search.default_similarity_threshold < 1 and
            self.search.default_top_k > 0 and
            self.search.max_query_variants > 0
        )
        
        # ?? Validate hybrid search settings
        validation_results["hybrid_search_config"] = bool(
            self.search.enable_hybrid_search and
            (self.search.enable_vector_search or self.search.enable_database_search)
        )
        
        return validation_results
    
    def get_entity_config(self, entity_name: str) -> Dict:
        """Get configuration for specific entity"""
        entity_lower = entity_name.lower()
        
        if entity_lower in self.entity_extraction.known_entities:
            return self.entity_extraction.known_entities[entity_lower]
        
        # Default configuration for unknown entities (hybrid-aware)
        return {
            "similarity_threshold": self.search.default_similarity_threshold,
            "top_k": self.search.default_top_k,
            "expected_docs": None,
            "search_strategy": "hybrid",      # ?? Default to hybrid
            "database_priority": False       # ?? Default no DB priority
        }
    
    def get_dynamic_search_params(self, query: str, extracted_entity: str = None) -> Dict:
        """Get dynamic search parameters based on query and entity with hybrid support"""
        query_lower = query.lower()
        
        # If we have extracted entity, use its configuration
        if extracted_entity:
            entity_config = self.get_entity_config(extracted_entity)
            return {
                "similarity_threshold": entity_config["similarity_threshold"],
                "top_k": entity_config["top_k"],
                "search_strategy": entity_config.get("search_strategy", "hybrid"),      # ??
                "database_priority": entity_config.get("database_priority", True),    # ??
                "enable_database_search": True                                          # ??
            }
        
        # Dynamic configuration based on query characteristics
        if len(query.split()) >= 4:  # Complex query
            return {
                "similarity_threshold": self.search.fallback_similarity_threshold,
                "top_k": self.search.complex_query_top_k,
                "search_strategy": "vector_priority",     # ??
                "database_priority": False,               # ??
                "enable_database_search": True            # ??
            }
        elif any(word in query_lower for word in ['tell', 'show', 'find', 'give']):  # Question format
            return {
                "similarity_threshold": self.search.entity_similarity_threshold,
                "top_k": self.search.entity_top_k,
                "search_strategy": "hybrid",              # ??
                "database_priority": True,                # ??
                "enable_database_search": True            # ??
            }
        else:  # Simple query
            return {
                "similarity_threshold": self.search.default_similarity_threshold,
                "top_k": self.search.default_top_k,
                "search_strategy": "hybrid",              # ??
                "database_priority": False,               # ??
                "enable_database_search": True            # ??
            }
    
    def get_search_strategy(self, query: str, extracted_entity: str = None) -> str:
        """?? Determine optimal search strategy for given query"""
        
        # Person name queries -> database priority
        if extracted_entity or any(word in query.lower() for word in ['who is', 'tell me about', 'show me']):
            return self.search.person_query_strategy
        
        # Complex queries -> vector priority
        if len(query.split()) >= 6:
            return self.search.general_query_strategy
        
        # Default -> hybrid
        return "hybrid"
    
    def is_person_query(self, query: str, extracted_entity: str = None) -> bool:
        """?? Detect if query is about a person"""
        if extracted_entity:
            return True
        
        person_indicators = ['who is', 'tell me about', 'show me', 'find', 'about']
        query_lower = query.lower()
        
        # Check for person indicators + capitalized words (likely names)
        has_person_indicator = any(indicator in query_lower for indicator in person_indicators)
        has_capitalized_words = bool([word for word in query.split() if word[0].isupper() and len(word) > 2])
        
        return has_person_indicator and has_capitalized_words

# Global configuration instance
config = ProductionRAGConfig()