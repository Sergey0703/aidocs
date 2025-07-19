#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module for RAG Document Indexer
Handles environment variables, validation, and default settings
"""

import os
from dotenv import load_dotenv


class Config:
    """Configuration class that loads and validates all settings"""
    
    def __init__(self):
        """Initialize configuration by loading environment variables"""
        load_dotenv()
        self._load_settings()
        self._validate_settings()
    
    def _load_settings(self):
        """Load all settings from environment variables with defaults"""
        
        # --- DIRECTORY AND FILE SETTINGS ---
        self.DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./data/634/2025")
        self.ERROR_LOG_FILE = "./indexing_errors.log"
        
        # --- DATABASE SETTINGS ---
        self.CONNECTION_STRING = os.getenv("SUPABASE_CONNECTION_STRING")
        self.TABLE_NAME = os.getenv("TABLE_NAME", "documents")
        
        # --- EMBEDDING SETTINGS ---
        self.EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
        self.EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # --- TEXT PROCESSING SETTINGS ---
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "256"))
        self.MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "100"))
        
        # --- BATCH PROCESSING SETTINGS ---
        self.PROCESSING_BATCH_SIZE = int(os.getenv("PROCESSING_BATCH_SIZE", "100"))
        self.EMBEDDING_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
        self.DB_BATCH_SIZE = int(os.getenv("DB_BATCH_SIZE", "25"))
        self.NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
        
        # --- OCR SETTINGS ---
        self.ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
        self.OCR_BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "10"))
        self.OCR_WORKERS = int(os.getenv("OCR_WORKERS", "4"))
        self.OCR_QUALITY_THRESHOLD = float(os.getenv("OCR_QUALITY_THRESHOLD", "0.3"))
        
        # --- PERFORMANCE SETTINGS ---
        self.OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
        self.MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        self.SKIP_VALIDATION = os.getenv("SKIP_VALIDATION", "false").lower() == "true"
    
    def _validate_settings(self):
        """Validate configuration settings and raise errors for critical issues"""
        
        # Critical validations
        if not self.CONNECTION_STRING:
            raise ValueError("SUPABASE_CONNECTION_STRING not found in .env file!")
        
        if not os.path.exists(self.DOCUMENTS_DIR):
            raise ValueError(f"Documents directory does not exist: {self.DOCUMENTS_DIR}")
        
        # Validate numeric ranges
        if self.CHUNK_SIZE < 100:
            raise ValueError("CHUNK_SIZE must be at least 100")
        
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        if self.EMBED_DIM not in [384, 512, 768, 1024, 1536]:
            print(f"WARNING: Unusual embedding dimension: {self.EMBED_DIM}")
        
        if self.PROCESSING_BATCH_SIZE < 1:
            raise ValueError("PROCESSING_BATCH_SIZE must be at least 1")
        
        if self.EMBEDDING_BATCH_SIZE < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE must be at least 1")
        
        if self.DB_BATCH_SIZE < 1:
            raise ValueError("DB_BATCH_SIZE must be at least 1")
    
    def print_config(self):
        """Print current configuration in a readable format"""
        print("=== ROBUST RAG INDEXER CONFIGURATION ===")
        print(f"Documents directory: {self.DOCUMENTS_DIR}")
        print(f"Embedding model: {self.EMBED_MODEL}")
        print(f"Chunk size: {self.CHUNK_SIZE}, Overlap: {self.CHUNK_OVERLAP}")
        print(f"Vector dimension: {self.EMBED_DIM}")
        print(f"Batch processing: {self.PROCESSING_BATCH_SIZE} chunks per batch")
        print(f"Error handling: Enhanced with encoding detection")
        print(f"Encoding support: UTF-8, Latin-1 (simplified for English files)")
        print(f"OCR enabled: {self.ENABLE_OCR}")
        print("=" * 60)
    
    def get_batch_settings(self):
        """Return batch processing settings as a dictionary"""
        return {
            'processing_batch_size': self.PROCESSING_BATCH_SIZE,
            'embedding_batch_size': self.EMBEDDING_BATCH_SIZE,
            'db_batch_size': self.DB_BATCH_SIZE,
            'num_workers': self.NUM_WORKERS
        }
    
    def get_chunk_settings(self):
        """Return text chunking settings as a dictionary"""
        return {
            'chunk_size': self.CHUNK_SIZE,
            'chunk_overlap': self.CHUNK_OVERLAP,
            'min_chunk_length': self.MIN_CHUNK_LENGTH
        }
    
    def get_embedding_settings(self):
        """Return embedding settings as a dictionary"""
        return {
            'model': self.EMBED_MODEL,
            'dimension': self.EMBED_DIM,
            'base_url': self.OLLAMA_BASE_URL,
            'timeout': self.OLLAMA_TIMEOUT
        }
    
    def get_ocr_settings(self):
        """Return OCR settings as a dictionary"""
        return {
            'enabled': self.ENABLE_OCR,
            'batch_size': self.OCR_BATCH_SIZE,
            'workers': self.OCR_WORKERS,
            'quality_threshold': self.OCR_QUALITY_THRESHOLD
        }


# Global configuration instance
config = Config()


def get_config():
    """Get the global configuration instance"""
    return config


def reload_config():
    """Reload configuration from environment variables"""
    global config
    config = Config()
    return config
