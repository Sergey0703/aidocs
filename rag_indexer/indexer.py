#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main RAG Document Indexer
Modular version with clean architecture, robust error handling, and end-to-end file analysis
"""

import logging
import sys
import time
from datetime import datetime

# --- LLAMA INDEX IMPORTS ---
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter

# --- LOCAL MODULES ---
from config import get_config
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


def initialize_components(config):
    """
    Initialize all LlamaIndex components
    
    Args:
        config: Configuration object
    
    Returns:
        dict: Initialized components
    """
    print("Initializing LlamaIndex components...")
    
    # Vector store
    vector_store = SupabaseVectorStore(
        postgres_connection_string=config.CONNECTION_STRING,
        collection_name=config.TABLE_NAME,
        dimension=config.EMBED_DIM,
    )
    
    # Storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Embedding model
    embed_model = OllamaEmbedding(
        model_name=config.EMBED_MODEL, 
        base_url=config.OLLAMA_BASE_URL
    )
    
    # Node parser
    node_parser = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP,
        paragraph_separator="\n\n",
        secondary_chunking_regex="[.!?]\\s+"
    )
    
    return {
        'vector_store': vector_store,
        'storage_context': storage_context,
        'embed_model': embed_model,
        'node_parser': node_parser
    }


def load_and_process_documents(config, progress_tracker):
    """
    Load and process all documents using simple approach
    
    Args:
        config: Configuration object
        progress_tracker: Progress tracker instance
    
    Returns:
        tuple: (text_documents, image_documents, basic_stats)
    """
    print(f"Loading documents from folder: {config.DOCUMENTS_DIR}")
    progress_tracker.add_checkpoint("Document loading started")
    
    # Load text documents with simple approach
    print("\n?? Loading text documents...")
    reader = create_safe_reader(
        config.DOCUMENTS_DIR, 
        recursive=True
    )
    
    # Simple loading - no complex tracking yet
    text_documents, basic_stats, _ = reader.load_data()
    
    progress_tracker.add_checkpoint("Text documents loaded", len(text_documents))
    
    # Load images with OCR
    image_documents = []
    if config.ENABLE_OCR:
        print("\n??? Processing images with OCR...")
        try:
            # Check OCR availability
            ocr_available, missing_libs = check_ocr_availability()
            if not ocr_available:
                print(f"WARNING: OCR libraries missing: {', '.join(missing_libs)}")
                print("Install with: pip install pytesseract pillow opencv-python")
            else:
                ocr_processor = create_ocr_processor(
                    quality_threshold=config.OCR_QUALITY_THRESHOLD,
                    batch_size=config.OCR_BATCH_SIZE
                )
                image_documents, ocr_stats = ocr_processor.process_images_in_directory(config.DOCUMENTS_DIR)
                progress_tracker.add_checkpoint("Image documents processed", len(image_documents))
        except Exception as e:
            print(f"WARNING: OCR processing failed: {e}")
            print("Continuing without OCR...")
    
    # Print basic loading summary
    reader.print_loading_summary()
    
    return text_documents, image_documents, basic_stats


def create_and_filter_chunks(documents, config, node_parser, progress_tracker):
    """
    Create and filter text chunks
    
    Args:
        documents: List of documents
        config: Configuration object
        node_parser: Node parser instance
        progress_tracker: Progress tracker instance
    
    Returns:
        tuple: (valid_nodes, invalid_nodes, node_stats)
    """
    print("\nCreating text chunks from all documents...")
    chunk_start_time = time.time()
    
    try:
        all_nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        progress_tracker.add_checkpoint("Chunks created", len(all_nodes))
    except Exception as e:
        print(f"ERROR: Failed to parse documents into chunks: {e}")
        raise
    
    chunk_time = time.time() - chunk_start_time
    print(f"Document chunking completed in {chunk_time:.2f} seconds")
    
    # Filter and enhance nodes
    node_processor = create_node_processor(config.MIN_CHUNK_LENGTH)
    valid_nodes, invalid_nodes = node_processor.filter_and_enhance_nodes(all_nodes)
    
    # Get node statistics
    node_stats = node_processor.get_node_statistics(valid_nodes)
    
    progress_tracker.add_checkpoint("Chunks filtered and enhanced", len(valid_nodes))
    
    print(f"Created {len(valid_nodes)} valid chunks for processing")
    if invalid_nodes:
        print(f"Filtered out {len(invalid_nodes)} invalid chunks")
    
    return valid_nodes, invalid_nodes, node_stats


def analyze_final_results(config, db_manager, log_dir):
    """
    Perform end-to-end analysis: compare directory with database to find missing files
    
    Args:
        config: Configuration object
        db_manager: Database manager instance
        log_dir: Directory for log files
    
    Returns:
        dict: Comprehensive analysis results
    """
    print(f"\n{'='*60}")
    print("END-TO-END FILE ANALYSIS")
    print(f"{'='*60}")
    
    # Perform comprehensive directory vs database comparison
    analysis_results = db_manager.analyze_directory_vs_database(
        config.DOCUMENTS_DIR, 
        recursive=True
    )
    
    # Extract results
    total_files = analysis_results['total_files_in_directory']
    files_in_db = analysis_results['files_successfully_in_db']
    missing_files = analysis_results['files_missing_from_db']
    missing_files_detailed = analysis_results['missing_files_detailed']
    success_rate = analysis_results['success_rate']
    
    print(f"\n?? Final Analysis Results:")
    print(f"  ?? Total files in directory: {total_files}")
    print(f"  ? Files successfully in database: {files_in_db}")
    print(f"  ? Files missing from database: {missing_files}")
    print(f"  ?? Success rate: {success_rate:.1f}%")
    
    # Save failed files details to log if any missing
    if missing_files_detailed:
        print(f"\n?? Saving {len(missing_files_detailed)} missing files details to log...")
        log_file_path = save_failed_files_details(missing_files_detailed, log_dir)
        if log_file_path:
            print(f"   ? Missing files details saved to: {log_file_path}")
        else:
            print(f"   ?? WARNING: Could not save missing files details")
        
        # Show first few missing files
        print(f"\n? First 5 missing files:")
        for i, missing_detail in enumerate(missing_files_detailed[:5]):
            print(f"  {i+1}. {missing_detail}")
        if len(missing_files_detailed) > 5:
            print(f"  ... and {len(missing_files_detailed) - 5} more (see detailed log)")
    else:
        print(f"\n? All files successfully processed - no missing files!")
    
    return analysis_results


def main():
    """Main function orchestrating the entire indexing process with end-to-end analysis"""
    
    # Setup
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Validate environment
    if not validate_python_version():
        sys.exit(1)
    
    # Initialize tracking
    progress_tracker = create_progress_tracker()
    performance_monitor = PerformanceMonitor()
    status_reporter = StatusReporter("RAG Document Indexing")
    
    # Setup logging
    log_dir = setup_logging_directory()
    
    start_time = time.time()
    progress_tracker.start()
    performance_monitor.start()
    
    # Statistics tracking
    stats = {
        'start_time': start_time,
        'documents_loaded': 0,
        'images_processed': 0,
        'chunks_created': 0,
        'embeddings_generated': 0,
        'records_saved': 0,
        'encoding_issues': 0
    }
    
    # Analysis results (will be filled at the end)
    final_analysis = None
    
    try:
        with InterruptHandler() as interrupt_handler:
            
            # ===============================================================
            # 1. LOAD CONFIGURATION
            # ===============================================================
            
            print("Loading configuration...")
            config = get_config()
            config.print_config()
            
            progress_tracker.add_checkpoint("Configuration loaded")
            
            # Print system information
            print_system_info()
            
            # Check OCR availability
            try:
                result = check_ocr_availability()
                if result and len(result) == 2:
                    ocr_available, missing_libs = result
                    if ocr_available:
                        print("OCR Status: Available")
                    else:
                        print(f"OCR Status: Missing libraries: {', '.join(missing_libs)}")
                        print("Install with: pip install pytesseract pillow opencv-python")
                else:
                    print("OCR Status: Check function returned unexpected result")
            except Exception as e:
                print(f"OCR Status: Error checking availability: {e}")
                print("OCR Status: Assuming not available")
            
            # ===============================================================
            # 2. INITIALIZE COMPONENTS
            # ===============================================================
            
            components = initialize_components(config)
            progress_tracker.add_checkpoint("Components initialized")
            
            # Create processors
            db_manager = create_database_manager(config.CONNECTION_STRING, config.TABLE_NAME)
            embedding_processor = create_embedding_processor(
                components['embed_model'], 
                components['vector_store']
            )
            batch_processor = create_batch_processor(
                embedding_processor, 
                config.PROCESSING_BATCH_SIZE
            )
            
            # Check for interruption
            if interrupt_handler.check_interrupted():
                print("Process interrupted during initialization")
                return
            
            # ===============================================================
            # 3. LOAD DOCUMENTS (SIMPLE APPROACH)
            # ===============================================================
            
            try:
                text_documents, image_documents, basic_stats = load_and_process_documents(
                    config, progress_tracker
                )
            except Exception as e:
                print(f"ERROR: Failed to load documents: {e}")
                raise
            
            # Combine documents
            documents = text_documents + image_documents
            stats['documents_loaded'] = len(text_documents)
            stats['images_processed'] = len(image_documents)
            
            load_time = time.time() - start_time
            print(f"\n?? Document Loading Results:")
            print(f"  ?? Time elapsed: {load_time:.2f} seconds")
            print(f"  ?? Text documents: {len(text_documents)}")
            print(f"  ??? Image documents: {len(image_documents)}")
            print(f"  ?? Total documents: {len(documents)}")
            
            if not documents:
                print("ERROR: No documents found in the specified directory.")
                return
            
            performance_monitor.checkpoint("Documents loaded", len(documents))
            
            # Check for interruption
            if interrupt_handler.check_interrupted():
                print("Process interrupted during document loading")
                return
            
            # ===============================================================
            # 4. SAFE DELETION DIALOG
            # ===============================================================
            
            print("\n" + "="*60)
            print("SAFE DELETION CHECK")
            print("="*60)
            
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
            progress_tracker.add_checkpoint("Deletion dialog completed")
            
            # Check for interruption
            if interrupt_handler.check_interrupted():
                print("Process interrupted during deletion dialog")
                return
            
            # ===============================================================
            # 5. CREATE AND FILTER CHUNKS
            # ===============================================================
            
            # Filter documents with content
            documents_with_content = []
            documents_without_content = []
            
            for doc in documents:
                file_name = doc.metadata.get('file_name', 'Unknown File')
                file_path = doc.metadata.get('file_path', file_name)
                text_content = doc.text.strip()
                
                if not text_content:
                    documents_without_content.append(f"{file_path} - EMPTY (no text extracted)")
                elif len(text_content) < config.MIN_CHUNK_LENGTH:
                    documents_without_content.append(f"{file_path} - TOO SHORT ({len(text_content)} chars)")
                else:
                    documents_with_content.append(doc)
            
            if documents_without_content:
                print(f"?? Found {len(documents_without_content)} documents without sufficient content.")
                print("   These documents loaded but contain inadequate text for processing.")
            
            if not documents_with_content:
                print("ERROR: No documents with sufficient text content found. Exiting.")
                return
            
            print(f"?? Processing {len(documents_with_content)} documents with valid content.")
            
            # Create chunks
            valid_nodes, invalid_nodes, node_stats = create_and_filter_chunks(
                documents_with_content, config, components['node_parser'], progress_tracker
            )
            
            stats['chunks_created'] = len(valid_nodes)
            
            if not valid_nodes:
                print("ERROR: No valid text chunks were generated. Exiting.")
                return
            
            performance_monitor.checkpoint("Chunks processed", len(valid_nodes))
            
            # Check for interruption
            if interrupt_handler.check_interrupted():
                print("Process interrupted during chunk creation")
                return
            
            # ===============================================================
            # 6. BATCH PROCESSING
            # ===============================================================
            
            print(f"\n?? Starting batch processing...")
            batch_settings = config.get_batch_settings()
            
            # Process all batches
            batch_results = batch_processor.process_all_batches(
                valid_nodes,
                batch_settings['embedding_batch_size'],
                batch_settings['db_batch_size']
            )
            
            # Update statistics
            stats['records_saved'] = batch_results['total_saved']
            stats['embeddings_generated'] = batch_results['total_saved']  # Approximate
            
            performance_monitor.checkpoint("Batch processing completed", batch_results['total_saved'])
            progress_tracker.add_checkpoint("Processing completed", batch_results['total_saved'])
            
            # ===============================================================
            # 7. END-TO-END ANALYSIS (NEW!)
            # ===============================================================
            
            # Perform comprehensive analysis AFTER everything is done
            final_analysis = analyze_final_results(config, db_manager, log_dir)
            
            # ===============================================================
            # 8. FINAL RESULTS AND CLEANUP
            # ===============================================================
            
            # Print final results
            success = batch_processor.print_final_results(batch_results, deletion_info)
            
            # Write comprehensive log
            batch_processor.write_comprehensive_log(
                batch_results, 
                deletion_info, 
                stats['encoding_issues'],
                f"{log_dir}/indexing_errors.log"
            )
            
            # Create and save run summary WITH END-TO-END ANALYSIS
            end_time = time.time()
            summary = create_run_summary(start_time, end_time, {
                **stats,
                **batch_results,
                'success': success
            }, final_analysis['missing_files_detailed'] if final_analysis else [])
            
            summary_file = f"{log_dir}/run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            if safe_file_write(summary_file, summary):
                print(f"?? Run summary saved to: {summary_file}")
            
            # Print performance summary
            performance_monitor.print_performance_summary()
            progress_tracker.print_progress_summary()
            
            # Final status report WITH END-TO-END ANALYSIS
            status_reporter.add_section("Final Statistics", {
                "Total processing time": f"{end_time - start_time:.2f}s",
                "Documents loaded": stats['documents_loaded'],
                "Images processed": stats['images_processed'],
                "Chunks created": stats['chunks_created'],
                "Records saved": stats['records_saved'],
                "Success rate": f"{batch_results['success_rate']:.1f}%",
                "Processing speed": f"{batch_results['avg_speed']:.2f} chunks/sec"
            })
            
            # END-TO-END ANALYSIS SECTION
            if final_analysis:
                status_reporter.add_section("End-to-End File Analysis", {
                    "Total files in directory": final_analysis['total_files_in_directory'],
                    "Files successfully in database": final_analysis['files_successfully_in_db'],
                    "Files missing from database": final_analysis['files_missing_from_db'],
                    "End-to-end success rate": f"{final_analysis['success_rate']:.1f}%",
                    "Missing files details": f"Saved to {log_dir}/failed_files_details.log" if final_analysis['missing_files_detailed'] else "No missing files"
                })
            
            status_reporter.add_section("Data Loss Analysis", {
                "Total chunks attempted": stats['chunks_created'],
                "Chunks successfully saved": stats['records_saved'],
                "Chunks lost": stats['chunks_created'] - stats['records_saved'],
                "Loss rate": f"{((stats['chunks_created'] - stats['records_saved']) / stats['chunks_created'] * 100):.2f}%" if stats['chunks_created'] > 0 else "0%",
                "Invalid chunks (filtered)": f"See invalid_chunks_report.log"
            })
            
            status_reporter.add_section("Quality Metrics", {
                "Processing pipeline errors": final_analysis['files_missing_from_db'] if final_analysis else 0,
                "Encoding issues": stats['encoding_issues'],
                "Failed chunks": batch_results['total_failed_chunks'],
                "Failed batches": batch_results['failed_batches'],
                "Embedding errors": batch_results['total_embedding_errors']
            })
            
            status_reporter.print_report()
            
            return success
    
    except KeyboardInterrupt:
        print("\n\n?? WARNING: Indexing interrupted by user.")
        if 'stats' in locals():
            print(f"?? INFO: Partial results: {stats.get('records_saved', 0)} chunks saved")
        print("?? INFO: No data was corrupted - safe to restart.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n? ERROR: FATAL ERROR: {e}")
        print("?? INFO: Check your configuration and try again.")
        
        # Try to save error information
        if 'log_dir' in locals():
            error_info = f"Fatal error at {datetime.now()}: {str(e)}\n"
            safe_file_write(f"{log_dir}/fatal_error.log", error_info)
        
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n?? WARNING: Indexing interrupted by user.")
        print("?? INFO: Safe to restart.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n? ERROR: FATAL ERROR: {e}")
        sys.exit(1)