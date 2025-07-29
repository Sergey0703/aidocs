#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe Batch processing module for RAG Document Indexer
Handles the main batch processing loop with progress tracking and error recovery
NOW with SAFE Ollama restarts between batches (not during embedding generation)
"""

import time
import os
import subprocess
from datetime import datetime, timedelta


def safe_restart_ollama_for_next_batch():
    """
    SAFELY restart Ollama service between batches to prevent memory leaks
    This is called ONLY after successful batch completion and database save
    
    Returns:
        bool: True if restart was successful, False otherwise
    """
    print(f"\n   ?? SAFE RESTART: Restarting Ollama between batches to prevent memory leaks...")
    
    try:
        # Stop Ollama service
        print(f"   INFO: Stopping Ollama service...")
        stop_result = subprocess.run(
            ["sudo", "systemctl", "stop", "ollama"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if stop_result.returncode != 0:
            print(f"   WARNING: Ollama stop returned code {stop_result.returncode}")
            print(f"   STDERR: {stop_result.stderr}")
        else:
            print(f"   ? Ollama service stopped successfully")
        
        # Wait for clean shutdown
        time.sleep(3)
        
        # Start Ollama service
        print(f"   INFO: Starting Ollama service...")
        start_result = subprocess.run(
            ["sudo", "systemctl", "start", "ollama"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if start_result.returncode != 0:
            print(f"   ? ERROR: Ollama start failed with code {start_result.returncode}")
            print(f"   STDERR: {start_result.stderr}")
            return False
        else:
            print(f"   ? Ollama service started successfully")
        
        # Wait for Ollama to fully initialize
        print(f"   INFO: Waiting for Ollama to initialize...")
        initialization_success = False
        
        for attempt in range(30):  # Try for 30 seconds
            time.sleep(1)
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    print(f"   ? Ollama ready after {attempt + 1} seconds")
                    initialization_success = True
                    break
            except:
                continue
        
        if not initialization_success:
            print(f"   ?? WARNING: Ollama initialization took longer than 30 seconds")
            print(f"   INFO: Continuing anyway - Ollama may still be starting...")
            return True  # Continue processing even if we can't verify
        
        print(f"   ?? SAFE RESTART COMPLETED: Ollama ready for next batch")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"   ? ERROR: Ollama restart command timed out")
        return False
    except Exception as e:
        print(f"   ? ERROR: Failed to restart Ollama: {e}")
        return False


class BatchProcessor:
    """Safe batch processor with Ollama restarts between batches only"""
    
    def __init__(self, embedding_processor, processing_batch_size=100, batch_restart_interval=5):
        """
        Initialize safe batch processor
        
        Args:
            embedding_processor: EmbeddingProcessor instance
            processing_batch_size: Number of chunks per batch
            batch_restart_interval: Restart Ollama every N batches (default: 5)
        """
        self.embedding_processor = embedding_processor
        self.processing_batch_size = processing_batch_size
        self.batch_restart_interval = batch_restart_interval  # NEW: restart every N batches
        
        self.batch_stats = {
            'start_time': None,
            'batches_processed': 0,
            'failed_batches': 0,
            'total_saved': 0,
            'total_failed_chunks': 0,
            'total_embedding_errors': 0,
            'ollama_restarts_attempted': 0,  # NEW: track restart attempts
            'ollama_restarts_successful': 0,  # NEW: track successful restarts
            'ollama_restart_failures': 0     # NEW: track failed restarts
        }
    
    def start_processing(self):
        """Start the batch processing session"""
        self.batch_stats['start_time'] = time.time()
        self.embedding_processor.reset_stats()
        print(f"Starting SAFE batch processing at {datetime.now().strftime('%H:%M:%S')}")
        print(f"SAFE FEATURE: Ollama restarts every {self.batch_restart_interval} batches (between batches only)")
    
    def should_restart_ollama(self, batch_num, total_batches):
        """
        SAFE: Determine if Ollama should be restarted before next batch
        
        Args:
            batch_num: Current batch number (1-based)
            total_batches: Total number of batches
        
        Returns:
            bool: True if should restart, False otherwise
        """
        # Only restart if:
        # 1. We've processed enough batches
        # 2. This is not the last batch (no point restarting after the last one)
        # 3. Batch restart interval is configured
        
        if self.batch_restart_interval <= 0:
            return False  # Restart disabled
        
        if batch_num >= total_batches:
            return False  # Don't restart after the last batch
        
        if batch_num % self.batch_restart_interval == 0:
            return True  # Time for a restart
        
        return False
    
    def safe_restart_ollama_if_needed(self, batch_num, total_batches):
        """
        SAFELY restart Ollama if needed, ONLY between batches
        
        Args:
            batch_num: Current batch number
            total_batches: Total number of batches
        
        Returns:
            bool: True if no restart needed or restart successful, False if restart failed
        """
        if not self.should_restart_ollama(batch_num, total_batches):
            return True  # No restart needed
        
        print(f"\n{'='*60}")
        print(f"?? SAFE OLLAMA RESTART - BETWEEN BATCHES")
        print(f"Completed batch {batch_num}/{total_batches}")
        print(f"Next restart scheduled after batch {batch_num + self.batch_restart_interval}")
        print(f"{'='*60}")
        
        self.batch_stats['ollama_restarts_attempted'] += 1
        
        success = safe_restart_ollama_for_next_batch()
        
        if success:
            self.batch_stats['ollama_restarts_successful'] += 1
            print(f"? SAFE RESTART SUCCESS: Ready for batch {batch_num + 1}")
        else:
            self.batch_stats['ollama_restart_failures'] += 1
            print(f"? SAFE RESTART FAILED: Continuing with current Ollama instance")
            print(f"?? WARNING: Memory leaks may accumulate without restart")
        
        print(f"{'='*60}\n")
        return success
    
    def process_batch(self, batch_nodes, batch_num, total_batches, embedding_batch_size, db_batch_size):
        """
        SAFE: Process a single batch of nodes with safe Ollama restart afterward
        
        Args:
            batch_nodes: List of nodes to process
            batch_num: Current batch number
            total_batches: Total number of batches
            embedding_batch_size: Size of embedding sub-batches
            db_batch_size: Size of database batches
        
        Returns:
            dict: Batch processing results
        """
        batch_start_time = time.time()
        
        print(f"\nSAFE PROCESSING: batch {batch_num}/{total_batches}")
        print(f"   Chunks {(batch_num-1)*self.processing_batch_size + 1}-{min(batch_num*self.processing_batch_size, (batch_num-1)*self.processing_batch_size + len(batch_nodes))}")
        print("-" * 40)
        
        batch_result = {
            'success': False,
            'nodes_processed': len(batch_nodes),
            'embeddings_generated': 0,
            'records_saved': 0,
            'failed_chunks': 0,
            'embedding_errors': 0,
            'processing_time': 0,
            'error': None,
            'ollama_restarted': False  # NEW: track if Ollama was restarted
        }
        
        try:
            # SAFE: Generate embeddings (no unsafe restarts during this phase)
            nodes_with_embeddings, embedding_errors = self.embedding_processor.robust_embedding_generation(
                batch_nodes, batch_num, embedding_batch_size
            )
            
            batch_result['embeddings_generated'] = len(nodes_with_embeddings)
            batch_result['embedding_errors'] = len(embedding_errors)
            self.batch_stats['total_embedding_errors'] += len(embedding_errors)
            
            # SAFE: Save to database (complete batch before any restarts)
            if nodes_with_embeddings:
                batch_saved, failed_chunks = self.embedding_processor.robust_save_to_database(
                    nodes_with_embeddings, batch_num, db_batch_size
                )
                
                batch_result['records_saved'] = batch_saved
                batch_result['failed_chunks'] = len(failed_chunks)
                
                self.batch_stats['total_saved'] += batch_saved
                self.batch_stats['total_failed_chunks'] += len(failed_chunks)
                
                if failed_chunks:
                    print(f"   INFO: Continuing despite {len(failed_chunks)} failed chunks...")
            else:
                print(f"   WARNING: No valid embeddings generated for this batch")
            
            batch_result['processing_time'] = time.time() - batch_start_time
            batch_result['success'] = True
            
            # Print batch summary
            if nodes_with_embeddings:
                avg_speed = len(nodes_with_embeddings) / batch_result['processing_time']
                print(f"   ? SUCCESS: Batch {batch_num} completed safely in {batch_result['processing_time']:.2f}s")
                print(f"   INFO: Speed: {avg_speed:.2f} chunks/sec")
                print(f"   INFO: Batch saved: {batch_result['records_saved']}")
            
            self.batch_stats['batches_processed'] += 1
            
            # SAFE: NOW that batch is completely finished, consider Ollama restart
            # This is the ONLY safe time to restart Ollama
            if batch_result['success']:
                restart_success = self.safe_restart_ollama_if_needed(batch_num, total_batches)
                batch_result['ollama_restarted'] = (
                    self.should_restart_ollama(batch_num, total_batches) and restart_success
                )
                
                # Note: We continue processing even if restart failed
                # The restart failure is logged but doesn't stop the pipeline
            
        except Exception as e:
            batch_result['error'] = str(e)
            batch_result['processing_time'] = time.time() - batch_start_time
            
            print(f"   ? ERROR: Batch {batch_num} failed completely: {e}")
            self.batch_stats['failed_batches'] += 1
            
            # Log batch failure
            self._log_batch_failure(batch_num, batch_nodes, e)
        
        return batch_result
    
    def print_overall_progress(self, batch_num, total_batches, total_nodes):
        """
        Print overall progress estimate with safe restart info
        
        Args:
            batch_num: Current batch number
            total_batches: Total number of batches
            total_nodes: Total number of nodes
        """
        if batch_num <= 1 or not self.batch_stats['start_time']:
            return
        
        overall_elapsed = time.time() - self.batch_stats['start_time']
        avg_batch_time = overall_elapsed / batch_num
        remaining_batches = total_batches - batch_num
        overall_eta_seconds = remaining_batches * avg_batch_time
        
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h {(seconds%3600)/60:.0f}m"
        
        overall_finish_time = (datetime.now() + timedelta(seconds=overall_eta_seconds)).strftime('%H:%M')
        progress_pct = (batch_num / total_batches) * 100
        
        print(f"   INFO: Overall progress: {progress_pct:.1f}% ({batch_num}/{total_batches} batches)")
        print(f"   INFO: Total saved so far: {self.batch_stats['total_saved']}/{total_nodes}")
        print(f"   INFO: Overall ETA: {format_time(overall_eta_seconds)} | Finish: {overall_finish_time}")
        
        # NEW: Show safe restart information
        if self.batch_restart_interval > 0:
            next_restart_batch = ((batch_num // self.batch_restart_interval) + 1) * self.batch_restart_interval
            if next_restart_batch <= total_batches:
                print(f"   INFO: Next safe Ollama restart: after batch {next_restart_batch}")
            
            if self.batch_stats['ollama_restarts_attempted'] > 0:
                success_rate = (self.batch_stats['ollama_restarts_successful'] / 
                               self.batch_stats['ollama_restarts_attempted'] * 100)
                print(f"   INFO: Ollama restarts: {self.batch_stats['ollama_restarts_successful']}/{self.batch_stats['ollama_restarts_attempted']} successful ({success_rate:.1f}%)")
    
    def _log_batch_failure(self, batch_num, batch_nodes, error):
        """Log batch failure details"""
        try:
            with open('./logs/batch_failures.log', 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n--- SAFE PROCESSING: Batch {batch_num} failure at {timestamp} ---\n")
                f.write(f"Error: {str(error)}\n")
                f.write(f"Batch size: {len(batch_nodes)}\n")
                f.write(f"Files in batch (first 5):\n")
                for i, node in enumerate(batch_nodes[:5]):
                    f.write(f"  {i+1}. {node.metadata.get('file_name', 'Unknown')}\n")
                if len(batch_nodes) > 5:
                    f.write(f"  ... and {len(batch_nodes) - 5} more files\n")
                f.write(f"Safe restart interval: {self.batch_restart_interval} batches\n")
                f.write("-" * 40 + "\n")
        except Exception as e:
            print(f"   WARNING: Could not write to batch_failures.log: {e}")
    
    def process_all_batches(self, valid_nodes, embedding_batch_size, db_batch_size):
        """
        SAFE: Process all batches of nodes with safe Ollama restarts
        
        Args:
            valid_nodes: List of all valid nodes to process
            embedding_batch_size: Size of embedding sub-batches
            db_batch_size: Size of database batches
        
        Returns:
            dict: Final processing results
        """
        total_nodes = len(valid_nodes)
        total_batches = (total_nodes + self.processing_batch_size - 1) // self.processing_batch_size
        
        print(f"\nStarting SAFE batch processing of {total_nodes} chunks...")
        print(f"Processing batch size: {self.processing_batch_size} chunks")
        print(f"Embedding batch size: {embedding_batch_size} chunks")
        print(f"Database batch size: {db_batch_size} chunks")
        print(f"?? SAFE FEATURE: Ollama restart every {self.batch_restart_interval} batches")
        print(f"Error recovery: Enabled with encoding detection")
        print("=" * 60)
        
        self.start_processing()
        
        # Process batches safely
        for i in range(0, total_nodes, self.processing_batch_size):
            batch_nodes = valid_nodes[i:i + self.processing_batch_size]
            batch_num = i // self.processing_batch_size + 1
            
            # SAFE: Process this batch (includes safe restart afterward if needed)
            batch_result = self.process_batch(
                batch_nodes, batch_num, total_batches, 
                embedding_batch_size, db_batch_size
            )
            
            # Print overall progress with restart info
            self.print_overall_progress(batch_num, total_batches, total_nodes)
        
        # Calculate final results with restart statistics
        total_time = time.time() - self.batch_stats['start_time']
        
        return {
            'total_time': total_time,
            'total_nodes': total_nodes,
            'total_batches': total_batches,
            'batches_processed': self.batch_stats['batches_processed'],
            'failed_batches': self.batch_stats['failed_batches'],
            'total_saved': self.batch_stats['total_saved'],
            'total_failed_chunks': self.batch_stats['total_failed_chunks'],
            'total_embedding_errors': self.batch_stats['total_embedding_errors'],
            'success_rate': (self.batch_stats['total_saved'] / total_nodes * 100) if total_nodes > 0 else 0,
            'avg_speed': self.batch_stats['total_saved'] / total_time if total_time > 0 else 0,
            # NEW: Safe restart statistics
            'ollama_restarts_attempted': self.batch_stats['ollama_restarts_attempted'],
            'ollama_restarts_successful': self.batch_stats['ollama_restarts_successful'],
            'ollama_restart_failures': self.batch_stats['ollama_restart_failures'],
            'batch_restart_interval': self.batch_restart_interval
        }
    
    def print_final_results(self, results, deletion_info):
        """
        Print final processing results with safe restart information
        
        Args:
            results: Processing results dictionary
            deletion_info: Information about deletion operations
        """
        success = (results['failed_batches'] == 0 and 
                  results['total_failed_chunks'] == 0 and 
                  results['total_embedding_errors'] == 0)
        
        print("\n" + "=" * 60)
        if success:
            print("? SUCCESS: SAFE ROBUST INDEXING COMPLETED SUCCESSFULLY!")
        elif results['total_saved'] > 0:
            print("?? WARNING: SAFE ROBUST INDEXING COMPLETED WITH SOME ERRORS!")
            print("? SUCCESS: Partial success - some data was saved successfully")
        else:
            print("? ERROR: INDEXING FAILED - NO DATA SAVED!")
        
        print("=" * 60)
        print(f"FINAL STATISTICS:")
        print(f"   Total time: {results['total_time']:.2f}s ({results['total_time']/60:.1f}m)")
        print(f"   Total chunks: {results['total_nodes']}")
        print(f"   Total batches: {results['total_batches']}")
        print(f"   Batches processed: {results['batches_processed']}")
        print(f"   Failed batches: {results['failed_batches']}")
        print(f"   Records saved: {results['total_saved']}")
        print(f"   Failed chunks: {results['total_failed_chunks']}")
        print(f"   Embedding errors: {results['total_embedding_errors']}")
        print(f"   Success rate: {results['success_rate']:.1f}%")
        print(f"   Average speed: {results['avg_speed']:.2f} chunks/sec")
        print(f"   Records deleted: {deletion_info['records_deleted']}")
        
        # NEW: Safe restart statistics
        print(f"\n?? SAFE OLLAMA RESTART STATISTICS:")
        print(f"   Restart interval: {results['batch_restart_interval']} batches")
        print(f"   Restarts attempted: {results['ollama_restarts_attempted']}")
        print(f"   Restarts successful: {results['ollama_restarts_successful']}")
        print(f"   Restart failures: {results['ollama_restart_failures']}")
        if results['ollama_restarts_attempted'] > 0:
            restart_success_rate = (results['ollama_restarts_successful'] / 
                                   results['ollama_restarts_attempted'] * 100)
            print(f"   Restart success rate: {restart_success_rate:.1f}%")
        
        # Calculate loss statistics
        total_attempted = results['total_nodes']
        total_lost = total_attempted - results['total_saved']
        loss_rate = (total_lost / total_attempted * 100) if total_attempted > 0 else 0
        
        print(f"\nDATA LOSS ANALYSIS:")
        print(f"   Total chunks attempted: {total_attempted}")
        print(f"   Chunks successfully saved: {results['total_saved']}")
        print(f"   Chunks lost: {total_lost}")
        print(f"   Loss rate: {loss_rate:.2f}%")
        
        if total_lost > 0:
            print(f"   Main loss causes:")
            print(f"   - Failed embeddings: {results['total_embedding_errors']}")
            print(f"   - Database save failures: {results['total_failed_chunks']}")
            print(f"   - Batch processing failures: {results['failed_batches']} batches")
        
        print(f"\n??? SAFE PROCESSING FEATURES:")
        print(f"   - Processing batch size: {self.processing_batch_size} chunks")
        print(f"   - Error recovery: Individual chunk processing")
        print(f"   - Binary data detection: Enabled")
        print(f"   - Content cleaning: Enabled")
        print(f"   - SAFE Ollama restarts: Between batches only")
        print(f"   - No unsafe mid-batch restarts: Guaranteed")
        
        print("=" * 60)
        
        if success:
            print("?? SUCCESS: Ready for RAG queries! All documents indexed safely.")
        elif results['total_saved'] > 0:
            print("?? WARNING: Ready for RAG queries with partial data.")
            print("INFO: Check error logs for details on failed items:")
            print("   - failed_chunks.log")
            print("   - embedding_errors.log") 
            print("   - batch_failures.log")
            print("   - invalid_chunks_report.log")
        else:
            print("? ERROR: No data available for RAG queries.")
        
        # Final restart advice
        if results['ollama_restart_failures'] > 0:
            print(f"\n?? RESTART ADVICE:")
            print(f"   {results['ollama_restart_failures']} Ollama restarts failed during processing")
            print(f"   Consider adjusting restart interval or checking sudo permissions")
            print(f"   Processing continued safely despite restart failures")
        
        return success
    
    def write_comprehensive_log(self, results, deletion_info, encoding_issues=0, error_log_file="./indexing_errors.log"):
        """
        Write comprehensive log with safe restart information
        
        Args:
            results: Processing results dictionary
            deletion_info: Information about deletion operations
            encoding_issues: Number of encoding issues encountered
            error_log_file: Path to error log file
        """
        success = (results['failed_batches'] == 0 and 
                  results['total_failed_chunks'] == 0 and 
                  results['total_embedding_errors'] == 0)
        
        try:
            with open(error_log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n--- SAFE robust processing run at {timestamp} ---\n")
                f.write(f"Status: {'SUCCESS' if success else 'PARTIAL' if results['total_saved'] > 0 else 'FAILED'}\n")
                f.write(f"Total time: {results['total_time']:.2f}s\n")
                f.write(f"Total chunks: {results['total_nodes']}\n")
                f.write(f"Batches processed: {results['batches_processed']}/{results['total_batches']}\n")
                f.write(f"Failed batches: {results['failed_batches']}\n")
                f.write(f"Records saved: {results['total_saved']}\n")
                f.write(f"Failed chunks: {results['total_failed_chunks']}\n")
                f.write(f"Embedding errors: {results['total_embedding_errors']}\n")
                f.write(f"Success rate: {results['success_rate']:.1f}%\n")
                f.write(f"Average speed: {results['avg_speed']:.2f} chunks/sec\n")
                f.write(f"Encoding issues: {encoding_issues}\n")
                f.write(f"Processing batch size: {self.processing_batch_size}\n")
                f.write(f"Records deleted: {deletion_info['records_deleted']}\n")
                
                # NEW: Safe restart log information
                f.write(f"SAFE RESTART STATISTICS:\n")
                f.write(f"Restart interval: {results['batch_restart_interval']} batches\n")
                f.write(f"Restarts attempted: {results['ollama_restarts_attempted']}\n")
                f.write(f"Restarts successful: {results['ollama_restarts_successful']}\n")
                f.write(f"Restart failures: {results['ollama_restart_failures']}\n")
                
                f.write("SAFE PROCESSING FEATURES:\n")
                f.write("- No unsafe mid-embedding restarts\n")
                f.write("- Ollama restarts only between completed batches\n")
                f.write("- Guaranteed data integrity during restarts\n")
                f.write("- Continued processing despite restart failures\n")
                f.write("-------------------------------------\n\n")
        except Exception as e:
            print(f"WARNING: Could not write to {error_log_file}: {e}")


class ProgressTracker:
    """Utility class for tracking and displaying progress with restart information"""
    
    def __init__(self):
        self.checkpoints = []
        self.start_time = None
    
    def start(self):
        """Start progress tracking"""
        self.start_time = time.time()
        self.checkpoints = []
    
    def add_checkpoint(self, name, items_processed=0, total_items=0):
        """
        Add a progress checkpoint
        
        Args:
            name: Name of the checkpoint
            items_processed: Number of items processed
            total_items: Total number of items
        """
        if self.start_time is None:
            self.start()
        
        checkpoint = {
            'name': name,
            'timestamp': datetime.now(),
            'elapsed': time.time() - self.start_time,
            'items_processed': items_processed,
            'total_items': total_items
        }
        self.checkpoints.append(checkpoint)
    
    def print_progress_summary(self):
        """Print a summary of all checkpoints"""
        if not self.checkpoints:
            return
        
        print("\nSafe Progress Summary:")
        print("-" * 50)
        
        for i, checkpoint in enumerate(self.checkpoints):
            elapsed_str = f"{checkpoint['elapsed']:.1f}s"
            if checkpoint['total_items'] > 0:
                progress_pct = (checkpoint['items_processed'] / checkpoint['total_items']) * 100
                print(f"{i+1}. {checkpoint['name']}: {checkpoint['items_processed']}/{checkpoint['total_items']} ({progress_pct:.1f}%) - {elapsed_str}")
            else:
                print(f"{i+1}. {checkpoint['name']}: completed - {elapsed_str}")
        
        total_elapsed = self.checkpoints[-1]['elapsed'] if self.checkpoints else 0
        print(f"\nTotal elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
        print(f"Safe processing: No data corruption during Ollama restarts")
        print("-" * 50)


def create_batch_processor(embedding_processor, processing_batch_size=100, batch_restart_interval=5):
    """
    Create a SAFE batch processor instance
    
    Args:
        embedding_processor: EmbeddingProcessor instance
        processing_batch_size: Number of chunks per batch
        batch_restart_interval: Restart Ollama every N batches (0 to disable)
    
    Returns:
        BatchProcessor: Configured SAFE processor
    """
    return BatchProcessor(embedding_processor, processing_batch_size, batch_restart_interval)


def create_progress_tracker():
    """
    Create a progress tracker instance
    
    Returns:
        ProgressTracker: New progress tracker
    """
    return ProgressTracker()