#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Configuration module for RAG Document Indexer
Handles environment variables, validation, and default settings
Added support for advanced document parsing, OCR improvements, and ENHANCED PDF PROCESSING
"""

import os
from dotenv import load_dotenv


class Config:
    """Enhanced configuration class with advanced document processing settings including PDF support"""
    
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
        
        # --- NEW: BACKUP AND BLACKLIST SETTINGS ---
        self.DOC_BACKUP_BASE_NAME = os.getenv("DOC_BACKUP_BASE_NAME", "doc_backups")
        
        # Blacklist directories that should be excluded from scanning
        blacklist_env = os.getenv("BLACKLIST_DIRECTORIES", "doc_backups,logs,temp,.git,__pycache__,.vscode,.idea,node_modules")
        self.BLACKLIST_DIRECTORIES = [dir.strip() for dir in blacklist_env.split(",") if dir.strip()]
        
        # --- DATABASE SETTINGS ---
        self.CONNECTION_STRING = os.getenv("SUPABASE_CONNECTION_STRING")
        self.TABLE_NAME = os.getenv("TABLE_NAME", "documents")
        
        # --- EMBEDDING SETTINGS ---
        self.EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # --- TEXT PROCESSING SETTINGS ---
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))
        self.MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "50"))
        
        # --- BATCH PROCESSING SETTINGS ---
        self.PROCESSING_BATCH_SIZE = int(os.getenv("PROCESSING_BATCH_SIZE", "50"))
        self.EMBEDDING_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
        self.DB_BATCH_SIZE = int(os.getenv("DB_BATCH_SIZE", "200"))
        self.NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
        
        # --- ENHANCED OCR SETTINGS ---
        self.ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
        self.OCR_BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "10"))
        self.OCR_WORKERS = int(os.getenv("OCR_WORKERS", "4"))
        self.OCR_QUALITY_THRESHOLD = float(os.getenv("OCR_QUALITY_THRESHOLD", "0.3"))
        
        # --- NEW: OCR ROTATION DETECTION ---
        self.OCR_AUTO_ROTATION = os.getenv("OCR_AUTO_ROTATION", "true").lower() == "true"
        self.OCR_ROTATION_QUALITY_THRESHOLD = float(os.getenv("OCR_ROTATION_QUALITY_THRESHOLD", "0.1"))
        self.OCR_TEST_ALL_ROTATIONS = os.getenv("OCR_TEST_ALL_ROTATIONS", "false").lower() == "true"
        self.OCR_ROTATION_TIMEOUT = int(os.getenv("OCR_ROTATION_TIMEOUT", "30"))
        self.OCR_SKIP_ROTATION_FOR_GOOD_QUALITY = os.getenv("OCR_SKIP_ROTATION_FOR_GOOD_QUALITY", "true").lower() == "true"
        
        # --- NEW: TEXT QUALITY ANALYSIS ---
        self.ENABLE_TEXT_QUALITY_ANALYSIS = os.getenv("ENABLE_TEXT_QUALITY_ANALYSIS", "false").lower() == "false"
        self.TEXT_QUALITY_MIN_SCORE = float(os.getenv("TEXT_QUALITY_MIN_SCORE", "0.3"))
        self.TEXT_QUALITY_MIN_WORDS = int(os.getenv("TEXT_QUALITY_MIN_WORDS", "5"))
        self.TEXT_QUALITY_MAX_IDENTICAL_CHARS = int(os.getenv("TEXT_QUALITY_MAX_IDENTICAL_CHARS", "10"))
        self.TEXT_QUALITY_LANGUAGE = os.getenv("TEXT_QUALITY_LANGUAGE", "english")  # english, russian, auto
        
        # --- NEW: ADVANCED DOCUMENT PARSING ---
        self.ENABLE_ADVANCED_DOC_PARSING = os.getenv("ENABLE_ADVANCED_DOC_PARSING", "true").lower() == "true"
        self.EXTRACT_IMAGES_FROM_DOCS = os.getenv("EXTRACT_IMAGES_FROM_DOCS", "true").lower() == "true"
        self.PRESERVE_DOC_STRUCTURE = os.getenv("PRESERVE_DOC_STRUCTURE", "true").lower() == "true"
        self.DOC_EXTRACT_TABLES = os.getenv("DOC_EXTRACT_TABLES", "true").lower() == "true"
        self.DOC_EXTRACT_HEADERS = os.getenv("DOC_EXTRACT_HEADERS", "true").lower() == "true"
        
        # --- NEW: HYBRID PROCESSING SETTINGS ---
        self.HYBRID_TEXT_IMAGE_PROCESSING = os.getenv("HYBRID_TEXT_IMAGE_PROCESSING", "true").lower() == "true"
        self.COMBINE_TEXT_AND_OCR_RESULTS = os.getenv("COMBINE_TEXT_AND_OCR_RESULTS", "true").lower() == "true"
        self.IMAGE_EXTRACTION_QUALITY = os.getenv("IMAGE_EXTRACTION_QUALITY", "high")  # low, medium, high
        
        # --- NEW: ENHANCED PDF PROCESSING SETTINGS ---
        self.ENABLE_ENHANCED_PDF_PROCESSING = os.getenv("ENABLE_ENHANCED_PDF_PROCESSING", "true").lower() == "true"
        self.PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", "2048"))
        self.PDF_PRESERVE_STRUCTURE = os.getenv("PDF_PRESERVE_STRUCTURE", "true").lower() == "true"
        self.PDF_MIN_SECTION_LENGTH = int(os.getenv("PDF_MIN_SECTION_LENGTH", "200"))
        self.PDF_HEADER_DETECTION = os.getenv("PDF_HEADER_DETECTION", "true").lower() == "true"
        self.PDF_FOOTER_CLEANUP = os.getenv("PDF_FOOTER_CLEANUP", "true").lower() == "true"
        self.PDF_ENABLE_OCR_FALLBACK = os.getenv("PDF_ENABLE_OCR_FALLBACK", "true").lower() == "true"
        
        # --- NEW: PDF PROCESSING STRATEGY SETTINGS ---
        self.PDF_AUTO_METHOD_SELECTION = os.getenv("PDF_AUTO_METHOD_SELECTION", "true").lower() == "true"
        self.PDF_PREFER_PYMUPDF = os.getenv("PDF_PREFER_PYMUPDF", "true").lower() == "true"
        self.PDF_ENABLE_TABLE_EXTRACTION = os.getenv("PDF_ENABLE_TABLE_EXTRACTION", "true").lower() == "true"
        self.PDF_SCANNED_THRESHOLD = float(os.getenv("PDF_SCANNED_THRESHOLD", "0.1"))  # Text coverage threshold
        self.PDF_TABLE_DETECTION_THRESHOLD = float(os.getenv("PDF_TABLE_DETECTION_THRESHOLD", "0.3"))
        
        # --- NEW: PDF OCR FALLBACK SETTINGS ---
        self.PDF_OCR_DPI = int(os.getenv("PDF_OCR_DPI", "300"))
        self.PDF_OCR_IMAGE_FORMAT = os.getenv("PDF_OCR_IMAGE_FORMAT", "jpeg")  # jpeg, png
        self.PDF_OCR_MIN_TEXT_LENGTH = int(os.getenv("PDF_OCR_MIN_TEXT_LENGTH", "20"))
        self.PDF_OCR_TIMEOUT_PER_PAGE = int(os.getenv("PDF_OCR_TIMEOUT_PER_PAGE", "30"))
        
        # --- NEW: PDF QUALITY AND VALIDATION ---
        self.PDF_MIN_CONTENT_LENGTH = int(os.getenv("PDF_MIN_CONTENT_LENGTH", "20"))  # Minimum chars for valid PDF
        self.PDF_MAX_PAGES_FOR_QUICK_ANALYSIS = int(os.getenv("PDF_MAX_PAGES_FOR_QUICK_ANALYSIS", "3"))
        self.PDF_ENABLE_CONTENT_VALIDATION = os.getenv("PDF_ENABLE_CONTENT_VALIDATION", "true").lower() == "true"
        
        # --- PERFORMANCE SETTINGS ---
        self.OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
        self.MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        self.SKIP_VALIDATION = os.getenv("SKIP_VALIDATION", "false").lower() == "true"
        
        # --- NEW: CPU OPTIMIZATION SETTINGS ---
        self.OLLAMA_NUM_THREAD = int(os.getenv("OLLAMA_NUM_THREAD", "16"))
        self.OLLAMA_NUMA = os.getenv("OLLAMA_NUMA", "false").lower() == "true"
        self.OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "60s")
        self.OLLAMA_MAX_LOADED_MODELS = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "1"))
        
        # --- NEW: MONITORING AND LOGGING ---
        self.ENABLE_PROGRESS_LOGGING = os.getenv("ENABLE_PROGRESS_LOGGING", "true").lower() == "true"
        self.LOG_BATCH_TIMING = os.getenv("LOG_BATCH_TIMING", "true").lower() == "true"
        self.LOG_OCR_ROTATION_ATTEMPTS = os.getenv("LOG_OCR_ROTATION_ATTEMPTS", "false").lower() == "true"
        self.LOG_TEXT_QUALITY_SCORES = os.getenv("LOG_TEXT_QUALITY_SCORES", "false").lower() == "true"
        self.LOG_PDF_PROCESSING_DETAILS = os.getenv("LOG_PDF_PROCESSING_DETAILS", "true").lower() == "true"
        
        # --- DOCUMENT CONVERSION SETTINGS ---
        self.AUTO_CONVERT_DOC = os.getenv("AUTO_CONVERT_DOC", "true").lower() == "true"
        self.BACKUP_ORIGINAL_DOC = os.getenv("BACKUP_ORIGINAL_DOC", "true").lower() == "true"
        self.DELETE_ORIGINAL_DOC = os.getenv("DELETE_ORIGINAL_DOC", "false").lower() == "true"
        
        # --- BATCH RESTART SETTINGS ---
        self.BATCH_RESTART_INTERVAL = int(os.getenv("BATCH_RESTART_INTERVAL", "5"))
    
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
        
        # NEW: Validate OCR rotation settings
        if self.OCR_ROTATION_QUALITY_THRESHOLD < 0 or self.OCR_ROTATION_QUALITY_THRESHOLD > 1:
            raise ValueError("OCR_ROTATION_QUALITY_THRESHOLD must be between 0 and 1")
        
        if self.OCR_ROTATION_TIMEOUT < 10:
            print("WARNING: OCR_ROTATION_TIMEOUT is very low, may cause timeouts")
        
        # NEW: Validate text quality settings
        if self.TEXT_QUALITY_MIN_SCORE < 0 or self.TEXT_QUALITY_MIN_SCORE > 1:
            raise ValueError("TEXT_QUALITY_MIN_SCORE must be between 0 and 1")
        
        if self.TEXT_QUALITY_LANGUAGE not in ["english", "russian", "auto"]:
            print(f"WARNING: Unsupported TEXT_QUALITY_LANGUAGE: {self.TEXT_QUALITY_LANGUAGE}")
        
        # NEW: Validate image extraction quality
        if self.IMAGE_EXTRACTION_QUALITY not in ["low", "medium", "high"]:
            print(f"WARNING: Invalid IMAGE_EXTRACTION_QUALITY: {self.IMAGE_EXTRACTION_QUALITY}, using 'high'")
            self.IMAGE_EXTRACTION_QUALITY = "high"
        
        # NEW: Validate PDF settings
        if self.PDF_CHUNK_SIZE < 100:
            raise ValueError("PDF_CHUNK_SIZE must be at least 100")
        
        if self.PDF_MIN_SECTION_LENGTH < 50:
            print("WARNING: PDF_MIN_SECTION_LENGTH is very low")
        
        if self.PDF_SCANNED_THRESHOLD < 0 or self.PDF_SCANNED_THRESHOLD > 1:
            raise ValueError("PDF_SCANNED_THRESHOLD must be between 0 and 1")
        
        if self.PDF_TABLE_DETECTION_THRESHOLD < 0 or self.PDF_TABLE_DETECTION_THRESHOLD > 1:
            raise ValueError("PDF_TABLE_DETECTION_THRESHOLD must be between 0 and 1")
        
        if self.PDF_OCR_DPI < 150:
            print("WARNING: PDF_OCR_DPI is low, may result in poor OCR quality")
        
        if self.PDF_OCR_IMAGE_FORMAT not in ["jpeg", "png"]:
            print(f"WARNING: Invalid PDF_OCR_IMAGE_FORMAT: {self.PDF_OCR_IMAGE_FORMAT}, using 'jpeg'")
            self.PDF_OCR_IMAGE_FORMAT = "jpeg"
        
        if self.PDF_MIN_CONTENT_LENGTH < 10:
            print("WARNING: PDF_MIN_CONTENT_LENGTH is very low")
        
        # NEW: Validate backup settings
        if not self.DOC_BACKUP_BASE_NAME:
            raise ValueError("DOC_BACKUP_BASE_NAME cannot be empty")
        
        if len(self.BLACKLIST_DIRECTORIES) == 0:
            print("WARNING: No directories in blacklist - all directories will be scanned")
    
    def get_backup_directory(self):
        absolute_backup = os.getenv("DOC_BACKUP_ABSOLUTE_PATH")
        if absolute_backup:
          return absolute_backup
    
        from pathlib import Path
        documents_path = Path(self.DOCUMENTS_DIR)
        backup_path = documents_path.parent / self.DOC_BACKUP_BASE_NAME
        return str(backup_path)
    
    def is_blacklisted_directory(self, directory_path):
        """
        Check if a directory should be excluded from scanning
        
        Args:
            directory_path: Path to check
        
        Returns:
            bool: True if directory should be excluded
        """
        path_parts = str(directory_path).split(os.sep)
        return any(blacklist_dir in path_parts for blacklist_dir in self.BLACKLIST_DIRECTORIES)
    
    def print_config(self):
        """Print current configuration in a readable format"""
        print("=== ENHANCED RAG INDEXER CONFIGURATION ===")
        print(f"Documents directory: {self.DOCUMENTS_DIR}")
        print(f"Backup directory: {self.get_backup_directory()}")
        print(f"Blacklisted directories: {', '.join(self.BLACKLIST_DIRECTORIES)}")
        print(f"Embedding model: {self.EMBED_MODEL} (CPU-optimized)")
        print(f"Chunk size: {self.CHUNK_SIZE}, Overlap: {self.CHUNK_OVERLAP}")
        print(f"Vector dimension: {self.EMBED_DIM}")
        print(f"Batch processing: {self.PROCESSING_BATCH_SIZE} chunks per batch")
        print(f"Batch restart interval: {self.BATCH_RESTART_INTERVAL} batches")
        print(f"Enhanced features:")
        print(f"  - Advanced document parsing: {'?' if self.ENABLE_ADVANCED_DOC_PARSING else '?'}")
        print(f"  - Auto .doc conversion: {'?' if self.AUTO_CONVERT_DOC else '?'}")
        print(f"  - OCR auto-rotation: {'?' if self.OCR_AUTO_ROTATION else '?'}")
        print(f"  - Text quality analysis: {'?' if self.ENABLE_TEXT_QUALITY_ANALYSIS else '?'}")
        print(f"  - Hybrid text+image processing: {'?' if self.HYBRID_TEXT_IMAGE_PROCESSING else '?'}")
        print(f"  - Extract images from docs: {'?' if self.EXTRACT_IMAGES_FROM_DOCS else '?'}")
        print(f"  - Structure preservation: {'?' if self.PRESERVE_DOC_STRUCTURE else '?'}")
        print(f"  - Enhanced PDF processing: {'?' if self.ENABLE_ENHANCED_PDF_PROCESSING else '?'}")
        print(f"PDF processing settings:")
        print(f"  - Auto method selection: {'?' if self.PDF_AUTO_METHOD_SELECTION else '?'}")
        print(f"  - Table extraction: {'?' if self.PDF_ENABLE_TABLE_EXTRACTION else '?'}")
        print(f"  - OCR fallback: {'?' if self.PDF_ENABLE_OCR_FALLBACK else '?'}")
        print(f"  - Chunk size: {self.PDF_CHUNK_SIZE}")
        print(f"CPU optimization: {self.OLLAMA_NUM_THREAD} threads, NUMA {'enabled' if self.OLLAMA_NUMA else 'disabled'}")
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
            'timeout': self.OLLAMA_TIMEOUT,
            'num_thread': self.OLLAMA_NUM_THREAD,
            'numa': self.OLLAMA_NUMA,
            'keep_alive': self.OLLAMA_KEEP_ALIVE
        }
    
    def get_ocr_settings(self):
        """Return enhanced OCR settings as a dictionary"""
        return {
            'enabled': self.ENABLE_OCR,
            'batch_size': self.OCR_BATCH_SIZE,
            'workers': self.OCR_WORKERS,
            'quality_threshold': self.OCR_QUALITY_THRESHOLD,
            'auto_rotation': self.OCR_AUTO_ROTATION,
            'rotation_quality_threshold': self.OCR_ROTATION_QUALITY_THRESHOLD,
            'test_all_rotations': self.OCR_TEST_ALL_ROTATIONS,
            'rotation_timeout': self.OCR_ROTATION_TIMEOUT,
            'skip_rotation_for_good_quality': self.OCR_SKIP_ROTATION_FOR_GOOD_QUALITY
        }
    
    def get_text_quality_settings(self):
        """Return text quality analysis settings as a dictionary"""
        return {
            'enabled': self.ENABLE_TEXT_QUALITY_ANALYSIS,
            'min_score': self.TEXT_QUALITY_MIN_SCORE,
            'min_words': self.TEXT_QUALITY_MIN_WORDS,
            'max_identical_chars': self.TEXT_QUALITY_MAX_IDENTICAL_CHARS,
            'language': self.TEXT_QUALITY_LANGUAGE
        }
    
    def get_document_parsing_settings(self):
        """Return advanced document parsing settings as a dictionary"""
        return {
            'advanced_parsing_enabled': self.ENABLE_ADVANCED_DOC_PARSING,
            'extract_images': self.EXTRACT_IMAGES_FROM_DOCS,
            'preserve_structure': self.PRESERVE_DOC_STRUCTURE,
            'extract_tables': self.DOC_EXTRACT_TABLES,
            'extract_headers': self.DOC_EXTRACT_HEADERS,
            'hybrid_processing': self.HYBRID_TEXT_IMAGE_PROCESSING,
            'combine_results': self.COMBINE_TEXT_AND_OCR_RESULTS,
            'image_quality': self.IMAGE_EXTRACTION_QUALITY
        }
    
    def get_pdf_processing_settings(self):
        """
        NEW: Return enhanced PDF processing settings as a dictionary
        
        Returns:
            dict: PDF processing configuration
        """
        return {
            'enabled': self.ENABLE_ENHANCED_PDF_PROCESSING,
            'chunk_size': self.PDF_CHUNK_SIZE,
            'preserve_structure': self.PDF_PRESERVE_STRUCTURE,
            'min_section_length': self.PDF_MIN_SECTION_LENGTH,
            'header_detection': self.PDF_HEADER_DETECTION,
            'footer_cleanup': self.PDF_FOOTER_CLEANUP,
            'enable_ocr_fallback': self.PDF_ENABLE_OCR_FALLBACK,
            'auto_method_selection': self.PDF_AUTO_METHOD_SELECTION,
            'prefer_pymupdf': self.PDF_PREFER_PYMUPDF,
            'enable_table_extraction': self.PDF_ENABLE_TABLE_EXTRACTION,
            'scanned_threshold': self.PDF_SCANNED_THRESHOLD,
            'table_detection_threshold': self.PDF_TABLE_DETECTION_THRESHOLD,
            'ocr_dpi': self.PDF_OCR_DPI,
            'ocr_image_format': self.PDF_OCR_IMAGE_FORMAT,
            'ocr_min_text_length': self.PDF_OCR_MIN_TEXT_LENGTH,
            'ocr_timeout_per_page': self.PDF_OCR_TIMEOUT_PER_PAGE,
            'min_content_length': self.PDF_MIN_CONTENT_LENGTH,
            'max_pages_for_analysis': self.PDF_MAX_PAGES_FOR_QUICK_ANALYSIS,
            'enable_content_validation': self.PDF_ENABLE_CONTENT_VALIDATION
        }
    
    def get_document_conversion_settings(self):
        """Return document conversion settings as a dictionary"""
        return {
            'auto_convert_doc': self.AUTO_CONVERT_DOC,
            'backup_original_doc': self.BACKUP_ORIGINAL_DOC,
            'delete_original_doc': self.DELETE_ORIGINAL_DOC,
            'backup_directory': self.get_backup_directory(),
            'blacklist_directories': self.BLACKLIST_DIRECTORIES
        }
    
    def get_performance_settings(self):
        """Return performance optimization settings as a dictionary"""
        return {
            'max_file_size': self.MAX_FILE_SIZE,
            'skip_validation': self.SKIP_VALIDATION,
            'ollama_timeout': self.OLLAMA_TIMEOUT,
            'num_workers': self.NUM_WORKERS,
            'cpu_threads': self.OLLAMA_NUM_THREAD,
            'numa_enabled': self.OLLAMA_NUMA,
            'keep_alive': self.OLLAMA_KEEP_ALIVE,
            'max_loaded_models': self.OLLAMA_MAX_LOADED_MODELS
        }
    
    def get_logging_settings(self):
        """Return logging and monitoring settings as a dictionary"""
        return {
            'progress_logging': self.ENABLE_PROGRESS_LOGGING,
            'batch_timing': self.LOG_BATCH_TIMING,
            'ocr_rotation_attempts': self.LOG_OCR_ROTATION_ATTEMPTS,
            'text_quality_scores': self.LOG_TEXT_QUALITY_SCORES,
            'pdf_processing_details': self.LOG_PDF_PROCESSING_DETAILS
        }
    
    def is_feature_enabled(self, feature_name):
        """
        Check if a specific feature is enabled
        
        Args:
            feature_name: Name of the feature to check
        
        Returns:
            bool: True if feature is enabled
        """
        feature_map = {
            'ocr': self.ENABLE_OCR,
            'advanced_doc_parsing': self.ENABLE_ADVANCED_DOC_PARSING,
            'auto_rotation': self.OCR_AUTO_ROTATION,
            'text_quality_analysis': self.ENABLE_TEXT_QUALITY_ANALYSIS,
            'hybrid_processing': self.HYBRID_TEXT_IMAGE_PROCESSING,
            'image_extraction': self.EXTRACT_IMAGES_FROM_DOCS,
            'structure_preservation': self.PRESERVE_DOC_STRUCTURE,
            'progress_logging': self.ENABLE_PROGRESS_LOGGING,
            'auto_convert_doc': self.AUTO_CONVERT_DOC,
            'enhanced_pdf_processing': self.ENABLE_ENHANCED_PDF_PROCESSING,
            'pdf_auto_method_selection': self.PDF_AUTO_METHOD_SELECTION,
            'pdf_table_extraction': self.PDF_ENABLE_TABLE_EXTRACTION,
            'pdf_ocr_fallback': self.PDF_ENABLE_OCR_FALLBACK
        }
        
        return feature_map.get(feature_name, False)


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


def print_feature_status():
    """Print status of all enhanced features including PDF processing"""
    config = get_config()
    
    print("\n=== ENHANCED FEATURES STATUS ===")
    features = [
        ("Advanced Document Parsing", config.is_feature_enabled('advanced_doc_parsing')),
        ("Auto .doc Conversion", config.is_feature_enabled('auto_convert_doc')),
        ("OCR Processing", config.is_feature_enabled('ocr')),
        ("OCR Auto-Rotation", config.is_feature_enabled('auto_rotation')),
        ("Text Quality Analysis", config.is_feature_enabled('text_quality_analysis')),
        ("Hybrid Text+Image Processing", config.is_feature_enabled('hybrid_processing')),
        ("Image Extraction from Docs", config.is_feature_enabled('image_extraction')),
        ("Document Structure Preservation", config.is_feature_enabled('structure_preservation')),
        ("Enhanced PDF Processing", config.is_feature_enabled('enhanced_pdf_processing')),
        ("PDF Auto Method Selection", config.is_feature_enabled('pdf_auto_method_selection')),
        ("PDF Table Extraction", config.is_feature_enabled('pdf_table_extraction')),
        ("PDF OCR Fallback", config.is_feature_enabled('pdf_ocr_fallback')),
        ("Progress Logging", config.is_feature_enabled('progress_logging')),
    ]
    
    for feature_name, enabled in features:
        status = "? ENABLED" if enabled else "? DISABLED"
        print(f"  {feature_name:<35}: {status}")
    
    print(f"\n??? Directory Settings:")
    print(f"  Documents directory: {config.DOCUMENTS_DIR}")
    print(f"  Backup directory: {config.get_backup_directory()}")
    print(f"  Blacklisted directories: {', '.join(config.BLACKLIST_DIRECTORIES)}")
    
    print(f"\n?? PDF Processing Settings:")
    pdf_settings = config.get_pdf_processing_settings()
    if pdf_settings['enabled']:
        print(f"  Chunk size: {pdf_settings['chunk_size']}")
        print(f"  Auto method selection: {'?' if pdf_settings['auto_method_selection'] else '?'}")
        print(f"  Table extraction: {'?' if pdf_settings['enable_table_extraction'] else '?'}")
        print(f"  OCR fallback: {'?' if pdf_settings['enable_ocr_fallback'] else '?'}")
        print(f"  Structure preservation: {'?' if pdf_settings['preserve_structure'] else '?'}")
        print(f"  OCR DPI: {pdf_settings['ocr_dpi']}")
        print(f"  Min content length: {pdf_settings['min_content_length']} chars")
    else:
        print(f"  PDF Processing: ? DISABLED")
    
    print("=" * 50)


def get_pdf_processing_capabilities():
    """
    NEW: Get PDF processing capabilities information
    
    Returns:
        dict: PDF capabilities status
    """
    config = get_config()
    
    capabilities = {
        'enhanced_pdf_enabled': config.ENABLE_ENHANCED_PDF_PROCESSING,
        'auto_method_selection': config.PDF_AUTO_METHOD_SELECTION,
        'table_extraction': config.PDF_ENABLE_TABLE_EXTRACTION,
        'ocr_fallback': config.PDF_ENABLE_OCR_FALLBACK,
        'structure_preservation': config.PDF_PRESERVE_STRUCTURE,
        'content_validation': config.PDF_ENABLE_CONTENT_VALIDATION,
        'settings': config.get_pdf_processing_settings()
    }
    
    return capabilities


def print_pdf_configuration_summary():
    """
    NEW: Print detailed PDF processing configuration summary
    """
    config = get_config()
    pdf_settings = config.get_pdf_processing_settings()
    
    print("\n" + "=" * 60)
    print("?? ENHANCED PDF PROCESSING CONFIGURATION")
    print("=" * 60)
    
    if not pdf_settings['enabled']:
        print("? Enhanced PDF processing is DISABLED")
        print("Enable with: ENABLE_ENHANCED_PDF_PROCESSING=true")
        return
    
    print("? Enhanced PDF processing is ENABLED")
    print()
    
    print("?? Processing Strategy:")
    print(f"  Auto method selection: {'?' if pdf_settings['auto_method_selection'] else '?'}")
    print(f"  Prefer PyMuPDF: {'?' if pdf_settings['prefer_pymupdf'] else '?'}")
    print(f"  Scanned PDF threshold: {pdf_settings['scanned_threshold']}")
    print(f"  Table detection threshold: {pdf_settings['table_detection_threshold']}")
    
    print("\n?? Content Processing:")
    print(f"  Chunk size: {pdf_settings['chunk_size']} characters")
    print(f"  Min section length: {pdf_settings['min_section_length']} characters")
    print(f"  Min content length: {pdf_settings['min_content_length']} characters")
    print(f"  Preserve structure: {'?' if pdf_settings['preserve_structure'] else '?'}")
    
    print("\n?? Features:")
    print(f"  Table extraction: {'?' if pdf_settings['enable_table_extraction'] else '?'}")
    print(f"  Header detection: {'?' if pdf_settings['header_detection'] else '?'}")
    print(f"  Footer cleanup: {'?' if pdf_settings['footer_cleanup'] else '?'}")
    print(f"  Content validation: {'?' if pdf_settings['enable_content_validation'] else '?'}")
    
    print("\n?? OCR Fallback:")
    if pdf_settings['enable_ocr_fallback']:
        print("  ? OCR fallback enabled for scanned PDFs")
        print(f"  OCR DPI: {pdf_settings['ocr_dpi']}")
        print(f"  Image format: {pdf_settings['ocr_image_format'].upper()}")
        print(f"  Min text length: {pdf_settings['ocr_min_text_length']} characters")
        print(f"  Timeout per page: {pdf_settings['ocr_timeout_per_page']} seconds")
    else:
        print("  ? OCR fallback disabled")
    
    print("\n? Performance:")
    print(f"  Max pages for quick analysis: {pdf_settings['max_pages_for_analysis']}")
    print(f"  CPU threads: {config.OLLAMA_NUM_THREAD}")
    print(f"  Processing batch size: {config.PROCESSING_BATCH_SIZE}")
    
    print("=" * 60)


def validate_pdf_processing_environment():
    """
    NEW: Validate that PDF processing environment is properly configured
    
    Returns:
        dict: Validation results
    """
    config = get_config()
    validation = {
        'pdf_enabled': config.ENABLE_ENHANCED_PDF_PROCESSING,
        'missing_libraries': [],
        'configuration_issues': [],
        'warnings': [],
        'ready': False
    }
    
    if not config.ENABLE_ENHANCED_PDF_PROCESSING:
        validation['configuration_issues'].append("Enhanced PDF processing is disabled")
        return validation
    
    # Check for required libraries
    try:
        import fitz
    except ImportError:
        validation['missing_libraries'].append('PyMuPDF (pip install PyMuPDF)')
    
    try:
        import pdfplumber
    except ImportError:
        validation['missing_libraries'].append('pdfplumber (pip install pdfplumber)')
    
    if config.PDF_ENABLE_OCR_FALLBACK:
        try:
            from pdf2image import convert_from_path
        except ImportError:
            validation['missing_libraries'].append('pdf2image (pip install pdf2image)')
    
    # Check configuration values
    if config.PDF_CHUNK_SIZE < 100:
        validation['configuration_issues'].append(f"PDF_CHUNK_SIZE too small: {config.PDF_CHUNK_SIZE}")
    
    if config.PDF_MIN_CONTENT_LENGTH < 10:
        validation['warnings'].append(f"PDF_MIN_CONTENT_LENGTH very low: {config.PDF_MIN_CONTENT_LENGTH}")
    
    if config.PDF_OCR_DPI < 200:
        validation['warnings'].append(f"PDF_OCR_DPI may be too low for good OCR: {config.PDF_OCR_DPI}")
    
    # Check if OCR is available when OCR fallback is enabled
    if config.PDF_ENABLE_OCR_FALLBACK and not config.ENABLE_OCR:
        validation['warnings'].append("PDF OCR fallback enabled but main OCR is disabled")
    
    # Determine readiness
    validation['ready'] = (
        len(validation['missing_libraries']) == 0 and
        len(validation['configuration_issues']) == 0
    )
    
    return validation


def print_pdf_environment_status():
    """
    NEW: Print PDF processing environment status
    """
    validation = validate_pdf_processing_environment()
    
    print("\n" + "=" * 60)
    print("?? PDF PROCESSING ENVIRONMENT STATUS")
    print("=" * 60)
    
    if validation['ready']:
        print("? PDF processing environment is READY")
    else:
        print("? PDF processing environment has ISSUES")
    
    if validation['missing_libraries']:
        print("\n?? Missing Libraries:")
        for lib in validation['missing_libraries']:
            print(f"  ? {lib}")
        print("\nInstall missing libraries to enable full PDF processing capabilities.")
    
    if validation['configuration_issues']:
        print("\n?? Configuration Issues:")
        for issue in validation['configuration_issues']:
            print(f"  ? {issue}")
    
    if validation['warnings']:
        print("\n?? Warnings:")
        for warning in validation['warnings']:
            print(f"  ?? {warning}")
    
    if not validation['missing_libraries'] and not validation['configuration_issues']:
        print("\n? All PDF processing libraries are available")
        print("? Configuration is valid")
        if validation['warnings']:
            print("?? Some warnings present but processing will work")
    
    print("=" * 60)


def get_recommended_pdf_env_vars():
    """
    NEW: Get recommended environment variables for PDF processing
    
    Returns:
        dict: Recommended .env settings
    """
    return {
        # Core PDF settings
        'ENABLE_ENHANCED_PDF_PROCESSING': 'true',
        'PDF_CHUNK_SIZE': '2048',
        'PDF_PRESERVE_STRUCTURE': 'true',
        'PDF_MIN_SECTION_LENGTH': '200',
        
        # Processing strategy
        'PDF_AUTO_METHOD_SELECTION': 'true',
        'PDF_PREFER_PYMUPDF': 'true',
        'PDF_ENABLE_TABLE_EXTRACTION': 'true',
        'PDF_SCANNED_THRESHOLD': '0.1',
        'PDF_TABLE_DETECTION_THRESHOLD': '0.3',
        
        # OCR fallback
        'PDF_ENABLE_OCR_FALLBACK': 'true',
        'PDF_OCR_DPI': '300',
        'PDF_OCR_IMAGE_FORMAT': 'jpeg',
        'PDF_OCR_MIN_TEXT_LENGTH': '20',
        'PDF_OCR_TIMEOUT_PER_PAGE': '30',
        
        # Quality and validation
        'PDF_MIN_CONTENT_LENGTH': '20',
        'PDF_MAX_PAGES_FOR_QUICK_ANALYSIS': '3',
        'PDF_ENABLE_CONTENT_VALIDATION': 'true',
        'PDF_HEADER_DETECTION': 'true',
        'PDF_FOOTER_CLEANUP': 'true',
        
        # Logging
        'LOG_PDF_PROCESSING_DETAILS': 'true'
    }


def print_pdf_env_recommendations():
    """
    NEW: Print recommended environment variable settings for PDF processing
    """
    recommended = get_recommended_pdf_env_vars()
    
    print("\n" + "=" * 60)
    print("?? RECOMMENDED PDF PROCESSING .ENV SETTINGS")
    print("=" * 60)
    print("Add these to your .env file for optimal PDF processing:")
    print()
    
    # Group settings by category
    categories = {
        "Core PDF Settings": [
            'ENABLE_ENHANCED_PDF_PROCESSING',
            'PDF_CHUNK_SIZE',
            'PDF_PRESERVE_STRUCTURE',
            'PDF_MIN_SECTION_LENGTH'
        ],
        "Processing Strategy": [
            'PDF_AUTO_METHOD_SELECTION',
            'PDF_PREFER_PYMUPDF',
            'PDF_ENABLE_TABLE_EXTRACTION',
            'PDF_SCANNED_THRESHOLD',
            'PDF_TABLE_DETECTION_THRESHOLD'
        ],
        "OCR Fallback": [
            'PDF_ENABLE_OCR_FALLBACK',
            'PDF_OCR_DPI',
            'PDF_OCR_IMAGE_FORMAT',
            'PDF_OCR_MIN_TEXT_LENGTH',
            'PDF_OCR_TIMEOUT_PER_PAGE'
        ],
        "Quality & Validation": [
            'PDF_MIN_CONTENT_LENGTH',
            'PDF_MAX_PAGES_FOR_QUICK_ANALYSIS',
            'PDF_ENABLE_CONTENT_VALIDATION',
            'PDF_HEADER_DETECTION',
            'PDF_FOOTER_CLEANUP'
        ],
        "Logging": [
            'LOG_PDF_PROCESSING_DETAILS'
        ]
    }
    
    for category, vars_list in categories.items():
        print(f"# {category}")
        for var in vars_list:
            if var in recommended:
                print(f"{var}={recommended[var]}")
        print()
    
    print("=" * 60)
    print("?? Tip: Copy these settings to your .env file and restart the application")
    print("=" * 60)


if __name__ == "__main__":
    # Test configuration when run directly
    print("?? Enhanced RAG Indexer Configuration Test")
    print("=" * 60)
    
    try:
        config = get_config()
        print("? Configuration loaded successfully")
        
        # Print feature status
        print_feature_status()
        
        # Print PDF configuration
        print_pdf_configuration_summary()
        
        # Check PDF environment
        print_pdf_environment_status()
        
        # Show recommendations
        print_pdf_env_recommendations()
        
    except Exception as e:
        print(f"? Configuration error: {e}")
        print("Check your .env file and fix any issues")