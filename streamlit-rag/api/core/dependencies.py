# api/core/dependencies.py
# Dependency injection for FastAPI routes

import logging
from typing import Dict
from fastapi import Depends

logger = logging.getLogger(__name__)


class SystemComponents:
    """Container for system components with singleton pattern"""
    
    _instance = None
    _components = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self):
        """Initialize all system components"""
        if self._components is not None:
            return  # Already initialized
        
        try:
            from config.settings import config
            from query_processing.entity_extractor import ProductionEntityExtractor
            from query_processing.query_rewriter import ProductionQueryRewriter
            from retrieval.multi_retriever import MultiStrategyRetriever
            from retrieval.results_fusion import HybridResultsFusionEngine
            
            logger.info("Initializing Production RAG System with Gemini API...")
            
            # Initialize components
            entity_extractor = ProductionEntityExtractor(config)
            query_rewriter = ProductionQueryRewriter(config)
            retriever = MultiStrategyRetriever(config)
            fusion_engine = HybridResultsFusionEngine(config)
            
            self._components = {
                "entity_extractor": entity_extractor,
                "query_rewriter": query_rewriter,
                "retriever": retriever,
                "fusion_engine": fusion_engine,
                "config": config
            }
            
            logger.info("Production RAG System initialized successfully with Gemini API")
            
        except Exception as e:
            logger.error(f"Failed to initialize system components: {e}")
            raise
    
    def get_components(self) -> Dict:
        """Get initialized components"""
        if self._components is None:
            self.initialize()
        return self._components


# Singleton instance
_system_components = SystemComponents()


def get_system_components() -> SystemComponents:
    """FastAPI dependency for system components"""
    return _system_components


def initialize_system_components():
    """Initialize system components at startup"""
    _system_components.initialize()