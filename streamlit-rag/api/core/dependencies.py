# api/core/dependencies.py
# Dependency injection for FastAPI - initialize system components once

import logging
from typing import Dict
from functools import lru_cache

from config.settings import config
from query_processing.entity_extractor import ProductionEntityExtractor
from query_processing.query_rewriter import ProductionQueryRewriter
from retrieval.multi_retriever import MultiStrategyRetriever
from retrieval.results_fusion import ResultsFusionEngine

logger = logging.getLogger(__name__)


class SystemComponents:
    """Singleton container for system components"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.components = None
            self._initialized = True
    
    def initialize(self) -> Dict:
        """Initialize all system components"""
        if self.components is not None:
            return self.components
        
        try:
            logger.info("Initializing Production RAG System with Gemini API...")
            
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
            
            self.components = {
                "entity_extractor": entity_extractor,
                "query_rewriter": query_rewriter,
                "retriever": retriever,
                "fusion_engine": fusion_engine,
                "status": component_status
            }
            
            logger.info("Production RAG System initialized successfully with Gemini API")
            return self.components
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    def get_components(self) -> Dict:
        """Get initialized components"""
        if self.components is None:
            return self.initialize()
        return self.components


@lru_cache()
def get_system_components() -> SystemComponents:
    """FastAPI dependency to get system components"""
    return SystemComponents()