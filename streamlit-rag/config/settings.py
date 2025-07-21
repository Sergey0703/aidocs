# config/settings.py
# Configuration settings for Production RAG System

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
    """Search and retrieval configuration"""
    # Dynamic thresholds based on query type
    default_similarity_threshold: float = 0.35
    entity_similarity_threshold: float = 0.3
    fallback_similarity_threshold: float = 0.25
    
    # Dynamic top_k based on query complexity
    default_top_k: int = 20
    entity_top_k: int = 50
    complex_query_top_k: int = 30
    
    # Multi-query settings
    max_query_variants: int = 3
    enable_query_rewriting: bool = True
    enable_entity_extraction: bool = True
    enable_multi_retrieval: bool = True
    
    # Results fusion
    min_results_for_fusion: int = 2
    max_final_results: int = 15
    fusion_method: str = "weighted_score"  # weighted_score, rank_fusion, hybrid

@dataclass
class EntityExtractionConfig:
    """Entity extraction configuration"""
    extraction_methods: List[str] = None
    fallback_enabled: bool = True
    validation_enabled: bool = True
    
    # Known entities for special handling
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
            self.known_entities = {
                "john nolan": {
                    "similarity_threshold": 0.3,
                    "top_k": 50,
                    "expected_docs": 9
                },
                "breeda daly": {
                    "similarity_threshold": 0.3,
                    "top_k": 40,
                    "expected_docs": 3
                }
            }

@dataclass
class QueryRewriteConfig:
    """Query rewriting configuration"""
    enabled: bool = True
    max_rewrites: int = 3
    rewrite_strategies: List[str] = None
    
    # Rewrite prompts
    expand_query_prompt: str = """Generate {num_queries} different ways to search for information about this topic. Make each query more specific and focused.

Original query: {query}

Generate {num_queries} search variations:"""
    
    simplify_query_prompt: str = """Simplify this query to extract the core search terms while preserving the meaning.

Complex query: {query}

Simplified query:"""
    
    def __post_init__(self):
        if self.rewrite_strategies is None:
            self.rewrite_strategies = ["expand", "simplify", "rephrase"]

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
    
    # Example queries
    example_queries: List[str] = None
    
    def __post_init__(self):
        if self.example_queries is None:
            self.example_queries = [
                "John Nolan",
                "tell me about John Nolan",
                "show me John Nolan certifications", 
                "who is Breeda Daly",
                "find Breeda Daly training",
                "what certifications does John Nolan have?",
                "give me information about Breeda Daly's courses"
            ]

class ProductionRAGConfig:
    """Main configuration class for Production RAG System"""
    
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
        
        return validation_results
    
    def get_entity_config(self, entity_name: str) -> Dict:
        """Get configuration for specific entity"""
        entity_lower = entity_name.lower()
        
        if entity_lower in self.entity_extraction.known_entities:
            return self.entity_extraction.known_entities[entity_lower]
        
        # Default configuration for unknown entities
        return {
            "similarity_threshold": self.search.default_similarity_threshold,
            "top_k": self.search.default_top_k,
            "expected_docs": None
        }
    
    def get_dynamic_search_params(self, query: str, extracted_entity: str = None) -> Dict:
        """Get dynamic search parameters based on query and entity"""
        query_lower = query.lower()
        
        # If we have extracted entity, use its configuration
        if extracted_entity:
            entity_config = self.get_entity_config(extracted_entity)
            return {
                "similarity_threshold": entity_config["similarity_threshold"],
                "top_k": entity_config["top_k"]
            }
        
        # Dynamic configuration based on query characteristics
        if len(query.split()) >= 4:  # Complex query
            return {
                "similarity_threshold": self.search.fallback_similarity_threshold,
                "top_k": self.search.complex_query_top_k
            }
        elif any(word in query_lower for word in ['tell', 'show', 'find', 'give']):  # Question format
            return {
                "similarity_threshold": self.search.entity_similarity_threshold,
                "top_k": self.search.entity_top_k
            }
        else:  # Simple query
            return {
                "similarity_threshold": self.search.default_similarity_threshold,
                "top_k": self.search.default_top_k
            }

# Global configuration instance
config = ProductionRAGConfig()