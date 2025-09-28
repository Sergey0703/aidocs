#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced RAG Document Indexer - Complete Integration
Enhanced version with advanced document parsing, auto-rotation OCR, and text quality analysis
Optimized for English documents with comprehensive error handling and progress tracking
UPDATED: Migrated from Ollama to Gemini API with gemini-embedding-001
"""

import logging
import sys
import time
from datetime import datetime

# --- LLAMA INDEX IMPORTS ---
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding  # UPDATED: Changed from Ollama to Gemini
from llama_index.core.node_parser import SentenceSplitter

# --- ENHANCED LOCAL MODULES ---
from config import get_config, print_feature_status
from file_utils import create_safe_reader
from ocr_processor import create_ocr_processor, check_ocr_availability
from database_manager import create_database_manager
from embedding_processor import create_embedding_processor, create_node_processor
from batch_processor import create_batch_processor, create_progress_tracker
from utils import (
    InterruptHandler, PerformanceMonitor, StatusReporter,
    validate_python_version, print_system_info, create_run_summary,
    setup_logging_directory, safe_file_write, save_failed_files_details
)

# --- HELPER MODULES ---
from loading_helpers import (
    load_and_process_documents_enhanced, 
    print_enhanced_loading_summary,
    validate_documents_for_processing,
    print_document_validation_summary
)
from analysis_helpers import (
    analyze_final_results_enhanced,
    create_enhanced_run_summary,
    create_enhanced_status_report
)
from chunk_helpers import (
    create_and_filter_chunks_enhanced,
    create_chunk_processing_report,
    save_chunk_processing_report
)


def print_advanced_parsing_info():
    """
    Print information about advanced parsing capabilities
    """
    print("\n🔧 Advanced Document Processing Features:")
    print("  📄 Automatic .doc to .docx conversion")
    print("  🔧 LibreOffice/Pandoc integration") 
    print("  💾 Safe file backup system")
    print("  📊 Enhanced text extraction")
    print("  🛡️ Robust error handling")
    print("  📈 Progress tracking")
    print("=" * 50)


def initialize_components(config):
    """
    Initialize all LlamaIndex components with enhanced settings and Gemini API
    
    Args:
        config: Enhanced configuration object
    
    Returns:
        dict: Initialized components
    """
    print("🔧 Initializing enhanced LlamaIndex components with Gemini API...")
    
    # Vector store with optimized settings
    vector_store = SupabaseVectorStore(
        postgres_connection_string=config.CONNECTION_STRING,
        collection_name=config.TABLE_NAME,
        dimension=config.EMBED_DIM,
    )
    
    # Storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # UPDATED: Gemini embedding model with API key authentication
    embed_settings = config.get_embedding_settings()
    embed_model = GeminiEmbedding(
        model_name=embed_settings['model'],
        api_key=embed_settings['api_key'],
        # NOTE: Gemini API doesn't use timeout in the same way as Ollama
        # Rate limiting is handled at the application level
    )
    
    # Enhanced node parser with optimized chunk sizes
    chunk_settings = config.get_chunk_settings()
    node_parser = SentenceSplitter(
        chunk_size=chunk_settings['chunk_size'], 
        chunk_overlap=chunk_settings['chunk_overlap'],
        paragraph_separator="\n\n",
        secondary_chunking_regex="[.!?]\\s+",
        include_metadata=True,
        include_prev_next_rel=True
    )
    
    print("✅ Enhanced components initialized successfully with Gemini API")
    return {
        'vector_store': vector_store,
        'storage_context': storage_context,
        'embed_model': embed_model,
        'node_parser': node_parser
    }


def main():
    """Enhanced main function with comprehensive processing and analysis using Gemini API"""
    
    # Setup
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Validate environment
    if not validate_python_version():
        sys.exit(1)
    
    # Initialize tracking
    progress_tracker = create_progress_tracker()
    performance_monitor = PerformanceMonitor()
    status_reporter = StatusReporter("Enhanced RAG Document Indexing (Gemini API)")
    
    # Setup logging
    log_dir = setup_logging_directory()
    
    start_time = time.time()
    progress_tracker.start()
    performance_monitor.start()
    
    # Enhanced statistics tracking
    stats = {
        'start_time': start_time,
        'documents_loaded': 0,
        'images_processed': 0,
        'chunks_created': 0,
        'valid_chunks': 0,
        'embeddings_generated': 0,
        'records_saved': 0,
        'encoding_issues': 0,
        'rotation_stats': {},
        'quality_analysis_results': {},
        'advanced_parsing_usage': 0,
        'processing_stages': []
    }
    
    # Final analysis results
    final_analysis = None
    
    try:
        with InterruptHandler() as interrupt_handler:
            
            # ===============================================================
            # 1. ENHANCED CONFIGURATION LOADING
            # ===============================================================
            
            print("🔧 Loading enhanced configuration...")
            config = get_config()
            config.print_config()
            
            # Print feature status
            print_feature_status()
            
            # Print advanced parsing info
            print_advanced_parsing_info()
            
            progress_tracker.add_checkpoint("Enhanced configuration loaded")
            
            # Print system information
            print_system_info()
            
            # UPDATED: Gemini API validation
            print("🚀 Validating Gemini API configuration...")
            from config import validate_gemini_environment, print_gemini_environment_status
            
            gemini_validation = validate_gemini_environment()
            if not gemini_validation['ready']:
                print("❌ Gemini API configuration issues detected:")
                for issue in gemini_validation['configuration_issues']:
                    print(f"   - {issue}")
                print("\nPlease fix configuration issues before proceeding.")
                sys.exit(1)
            else:
                print("✅ Gemini API configuration validated successfully")
                print(f"   Model: {config.EMBED_MODEL}")
                print(f"   Dimension: {config.EMBED_DIM}")
                print(f"   Rate limit: {config.GEMINI_REQUEST_RATE_LIMIT} requests/sec")
            
            # Enhanced OCR availability check
            if config.ENABLE_OCR:
                try:
                    ocr_available, missing_libs = check_ocr_availability()
                    if ocr_available:
                        print("🔍 OCR Status: ✅ All libraries available")
                        print(f"   Enhanced features: Auto-rotation, Quality analysis")
                    else:
                        print(f"🔍 OCR Status: ❌ Missing: {', '.join(missing_libs)}")
                        print("   Install with: pip install pytesseract pillow opencv-python")
                except Exception as e:
                    print(f"🔍 OCR Status: ⚠️ Error checking: {e}")
            else:
                print("🔍 OCR Status: ⚠️ Disabled in configuration")
            
            # ===============================================================
            # 2. ENHANCED COMPONENT INITIALIZATION
            # ===============================================================
            
            components = initialize_components(config)
            progress_tracker.add_checkpoint("Enhanced components initialized")
            
            # Create enhanced processors
            db_manager = create_database_manager(config.CONNECTION_STRING, config.TABLE_NAME)
            embedding_processor = create_embedding_processor(
                components['embed_model'], 
                components['vector_store']
            )
            
            # UPDATED: Create batch processor with Gemini-appropriate restart interval
            # For Gemini API, we don't need Ollama restarts, so set interval to 0
            batch_restart_interval = 0  # Gemini API doesn't need service restarts
            batch_processor = create_batch_processor(
                embedding_processor, 
                config.PROCESSING_BATCH_SIZE,
                batch_restart_interval
            )
            
            # Check for interruption
            if interrupt_handler.check_interrupted():
                print("Process interrupted during initialization")
                return
            
            # ===============================================================
            # 3. ENHANCED DOCUMENT LOADING
            # ===============================================================
            
            try:
                text_documents, image_documents, processing_summary = load_and_process_documents_enhanced(
                    config, progress_tracker
                )
                stats['processing_stages'].append('document_loading')
            except Exception as e:
                print(f"❌ Enhanced document loading failed: {e}")
                raise
            
            # Combine documents
            documents = text_documents + image_documents
            stats['documents_loaded'] = len(text_documents)
            stats['images_processed'] = len(image_documents)
            
            # Update stats with processing summary
            if processing_summary:
                if 'rotation_stats' in processing_summary:
                    stats['rotation_stats'] = processing_summary['rotation_stats']
            
            load_time = time.time() - start_time
            
            # Print enhanced loading summary
            print_enhanced_loading_summary(text_documents, image_documents, processing_summary, load_time)
            
            if not documents:
                print("⚠️ No documents found in the specified directory.")
                return
            
            performance_monitor.checkpoint("Enhanced documents loaded", len(documents))
            stats['processing_stages'].append('documents_combined')
            
            # Check for interruption
            if interrupt_handler.check_interrupted():
                print("Process interrupted during document loading")
                return
            
            # ===============================================================
            # 4. ENHANCED DELETION DIALOG
            # ===============================================================
            
            print(f"\n{'='*70}")
            print("🗑️ ENHANCED SAFE DELETION CHECK")
            print(f"{'='*70}")
            
            # Get file identifiers
            files_to_process = set()
            for doc in documents:
                file_path = doc.metadata.get('file_path', '')
                file_name = doc.metadata.get('file_name', '')
                if file_path:
                    files_to_process.add(file_path)
                elif file_name:
                    files_to_process.add(file_name)
            
            deletion_info = db_manager.safe_deletion_dialog(files_to_process)
            progress_tracker.add_checkpoint("Enhanced deletion dialog completed")
            stats['processing_stages'].append('deletion_dialog')
            
            # Check for interruption
            if interrupt_handler.check_interrupted():
                print("Process interrupted during deletion dialog")
                return
            
            # ===============================================================
            # 5. ENHANCED CHUNK CREATION AND FILTERING
            # ===============================================================
            
            # Validate documents first using helper function
            documents_with_content, documents_without_content = validate_documents_for_processing(documents, config)
            
            # Print validation summary
            print_document_validation_summary(documents_with_content, documents_without_content)
            
            if not documents_with_content:
                print("❌ No documents with sufficient text content found. Exiting.")
                return
            
            # Enhanced chunk creation and filtering using helper module
            valid_nodes, invalid_nodes, enhanced_node_stats = create_and_filter_chunks_enhanced(
                documents_with_content, config, components['node_parser'], progress_tracker
            )
            
            # Create comprehensive chunk processing report
            chunk_report = create_chunk_processing_report(valid_nodes, invalid_nodes, enhanced_node_stats, config)
            save_chunk_processing_report(chunk_report, log_dir)
            
            stats['chunks_created'] = enhanced_node_stats['total_nodes_created']
            stats['valid_chunks'] = enhanced_node_stats['valid_nodes']
            stats['quality_analysis_results'] = {
                'filter_success_rate': enhanced_node_stats['filter_success_rate'],
                'invalid_chunks': enhanced_node_stats['invalid_nodes'],
                'avg_content_length': enhanced_node_stats['avg_content_length']
            }
            stats['processing_stages'].append('chunk_processing')
            
            if not valid_nodes:
                print("❌ No valid text chunks were generated. Exiting.")
                return
            
            performance_monitor.checkpoint("Enhanced chunks processed", len(valid_nodes))
            
            # Check for interruption
            if interrupt_handler.check_interrupted():
                print("Process interrupted during chunk creation")
                return
            
            # ===============================================================
            # 6. ENHANCED BATCH PROCESSING WITH GEMINI API
            # ===============================================================
            
            print(f"\n🚀 Starting enhanced batch processing with Gemini API...")
            batch_settings = config.get_batch_settings()
            
            print(f"🔧 Enhanced Processing Configuration (Gemini API):")
            print(f"   Processing batch size: {batch_settings['processing_batch_size']}")
            print(f"   Embedding batch size: {batch_settings['embedding_batch_size']}")
            print(f"   Database batch size: {batch_settings['db_batch_size']}")
            print(f"   Embedding model: {config.EMBED_MODEL} ({config.EMBED_DIM}D)")
            print(f"   Gemini rate limit: {config.GEMINI_REQUEST_RATE_LIMIT} requests/sec")
            print(f"   Gemini retry attempts: {config.GEMINI_RETRY_ATTEMPTS}")
            print(f"   Gemini timeout: {config.GEMINI_TIMEOUT}s")
            print(f"   Service restarts: Not applicable for Gemini API")
            
            # Process all batches with enhanced monitoring
            batch_results = batch_processor.process_all_batches(
                valid_nodes,
                batch_settings['embedding_batch_size'],
                batch_settings['db_batch_size']
            )
            
            # Update enhanced statistics
            stats['records_saved'] = batch_results['total_saved']
            stats['embeddings_generated'] = batch_results['total_saved']  # Approximate
            stats['processing_stages'].append('batch_processing')
            
            # Add batch processing stats
            stats.update({
                'total_batches': batch_results['total_batches'],
                'failed_batches': batch_results['failed_batches'],
                'total_failed_chunks': batch_results['total_failed_chunks'],
                'total_embedding_errors': batch_results['total_embedding_errors'],
                'avg_speed': batch_results['avg_speed'],
                'total_time': batch_results['total_time']
            })
            
            performance_monitor.checkpoint("Enhanced batch processing completed", batch_results['total_saved'])
            progress_tracker.add_checkpoint("Enhanced processing completed", batch_results['total_saved'])
            
            # ===============================================================
            # 7. ENHANCED END-TO-END ANALYSIS
            # ===============================================================
            
            # Perform comprehensive enhanced analysis
            final_analysis = analyze_final_results_enhanced(config, db_manager, log_dir, stats)
            stats['processing_stages'].append('final_analysis')
            
            # ===============================================================
            # 8. ENHANCED FINAL RESULTS AND REPORTING
            # ===============================================================
            
            # Print enhanced final results
            success = batch_processor.print_final_results(batch_results, deletion_info)
            
            # Write comprehensive enhanced log
            batch_processor.write_comprehensive_log(
                batch_results, 
                deletion_info, 
                stats['encoding_issues'],
                f"{log_dir}/enhanced_indexing_errors.log"
            )
            
            # Create and save enhanced run summary
            end_time = time.time()
            enhanced_summary = create_enhanced_run_summary(
                start_time, end_time, stats, final_analysis, config
            )
            
            summary_file = f"{log_dir}/enhanced_run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            if safe_file_write(summary_file, enhanced_summary):
                print(f"📊 Enhanced run summary saved to: {summary_file}")
            
            # Print enhanced performance summary
            performance_monitor.print_performance_summary()
            progress_tracker.print_progress_summary()
            
            # Enhanced final status report
            create_enhanced_status_report(
                status_reporter, stats, final_analysis, batch_results, 
                deletion_info, start_time, end_time
            )
            
            status_reporter.print_report()
            
            return success
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️ WARNING: Enhanced indexing interrupted by user.")
        if 'stats' in locals():
            print(f"📊 Partial results: {stats.get('records_saved', 0)} chunks saved")
        print(f"✅ No data was corrupted - safe to restart.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n❌ ERROR: FATAL ERROR: {e}")
        print(f"🔧 Check your configuration and try again.")
        
        # Try to save error information
        if 'log_dir' in locals():
            error_info = f"Enhanced indexer fatal error at {datetime.now()}: {str(e)}\n"
            safe_file_write(f"{log_dir}/enhanced_fatal_error.log", error_info)
        
        sys.exit(1)


if __name__ == "__main__":
    try:
        print("🚀 Enhanced RAG Document Indexer (Gemini API)")
        print("=" * 50)
        print("✨ Advanced features: Auto .doc conversion, OCR auto-rotation, Gemini API embeddings")
        print("🎯 Optimized for English documents with comprehensive error handling")
        print("=" * 50)
        
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Enhanced indexing interrupted by user.")
        print(f"✅ Safe to restart - no data corruption.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        sys.exit(1)