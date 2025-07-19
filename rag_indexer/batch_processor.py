#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch processing module for RAG Document Indexer
Handles the main batch processing loop with progress tracking and error recovery
"""

import time
from datetime import datetime, timedelta


class BatchProcessor:
    """Main batch processor for handling document indexing workflow"""
    
    def __init__(self, embedding_processor, processing_batch_size=100):
        """
        Initialize batch processor
        
        Args:
            embedding_processor: EmbeddingProcessor instance
            processing_batch_size: Number of chunks per batch
        """
        self.embedding_processor = embedding_processor
        self.processing_batch_size = processing_batch_size
        self.batch_stats = {
            'start_time': None,
            'batches_processed': 0,
            'failed_batches': 0,
            'total_saved': 0,
            'total_failed_chunks': 0,
            'total_embedding_errors': 0
        }
    
    def start_processing(self):
        """Start the batch processing session"""
        self.batch_stats['start_time'] = time.time()
        self.embedding_processor.reset_stats()
        print(f"Starting batch processing at {datetime.now().strftime('%H:%M:%S')}")
    
    def process_batch(self, batch_nodes, batch_num, total_batches, embedding_batch_size, db_batch_size):
        """
        Process a single batch of nodes
        
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
        
        print(f"\nProcessing batch {batch_num}/{total_batches}")
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
            'error': None
        }
        
        try:
            # Generate embeddings
            nodes_with_embeddings, embedding_errors = self.embedding_processor.robust_embedding_generation(
                batch_nodes, batch_num, embedding_batch_size
            )
            
            batch_result['embeddings_generated'] = len(nodes_with_embeddings)
            batch_result['embedding_errors'] = len(embedding_errors)
            self.batch_stats['total_embedding_errors'] += len(embedding_errors)
            
            # Save to database
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
                print(f"   SUCCESS: Batch {batch_num} completed in {batch_result['processing_time']:.2f}s")
                print(f"   INFO: Speed: {avg_speed:.2f} chunks/sec")
                print(f"   INFO: Batch saved: {batch_result['records_saved']}")
            
            self.batch_stats['batches_processed'] += 1
            
        except Exception as e:
            batch_result['error'] = str(e)
            batch_result['processing_time'] = time.time() - batch_start_time
            
            print(f"   ERROR: Batch {batch_num} failed completely: {e}")
            self.batch_stats['failed_batches'] += 1
            
            # Log batch failure
            self._log_batch_failure(batch_num, batch_nodes, e)
        
        return batch_result
    
    def print_overall_progress(self, batch_num, total_batches, total_nodes):
        """
        Print overall progress estimate
        
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
    
    def _log_batch_failure(self, batch_num, batch_nodes, error):
        """Log batch failure details"""
        try:
            with open('./batch_failures.log', 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n--- Batch {batch_num} failure at {timestamp} ---\n")
                f.write(f"Error: {str(error)}\n")
                f.write(f"Batch size: {len(batch_nodes)}\n")
                f.write(f"Files in batch (first 5):\n")
                for i, node in enumerate(batch_nodes[:5]):
                    f.write(f"  {i+1}. {node.metadata.get('file_name', 'Unknown')}\n")
                if len(batch_nodes) > 5:
                    f.write(f"  ... and {len(batch_nodes) - 5} more files\n")
                f.write("-" * 40 + "\n")
        except Exception as e:
            print(f"   WARNING: Could not write to batch_failures.log: {e}")
    
    def process_all_batches(self, valid_nodes, embedding_batch_size, db_batch_size):
        """
        Process all batches of nodes
        
        Args:
            valid_nodes: List of all valid nodes to process
            embedding_batch_size: Size of embedding sub-batches
            db_batch_size: Size of database batches
        
        Returns:
            dict: Final processing results
        """
        total_nodes = len(valid_nodes)
        total_batches = (total_nodes + self.processing_batch_size - 1) // self.processing_batch_size
        
        print(f"\nStarting robust batch processing of {total_nodes} chunks...")
        print(f"Processing batch size: {self.processing_batch_size} chunks")
        print(f"Embedding batch size: {embedding_batch_size} chunks")
        print(f"Database batch size: {db_batch_size} chunks")
        print(f"Error recovery: Enabled with encoding detection")
        print("=" * 60)
        
        self.start_processing()
        
        # Process batches
        for i in range(0, total_nodes, self.processing_batch_size):
            batch_nodes = valid_nodes[i:i + self.processing_batch_size]
            batch_num = i // self.processing_batch_size + 1
            
            # Process this batch
            batch_result = self.process_batch(
                batch_nodes, batch_num, total_batches, 
                embedding_batch_size, db_batch_size
            )
            
            # Print overall progress
            self.print_overall_progress(batch_num, total_batches, total_nodes)
        
        # Calculate final results
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
            'avg_speed': self.batch_stats['total_saved'] / total_time if total_time > 0 else 0
        }
    
    def print_final_results(self, results, deletion_info):
        """
        Print final processing results
        
        Args:
            results: Processing results dictionary
            deletion_info: Information about deletion operations
        """
        success = (results['failed_batches'] == 0 and 
                  results['total_failed_chunks'] == 0 and 
                  results['total_embedding_errors'] == 0)
        
        print("\n" + "=" * 60)
        if success:
            print("SUCCESS: ROBUST INDEXING COMPLETED SUCCESSFULLY!")
        elif results['total_saved'] > 0:
            print("WARNING: ROBUST INDEXING COMPLETED WITH SOME ERRORS!")
            print("SUCCESS: Partial success - some data was saved successfully")
        else:
            print("ERROR: INDEXING FAILED - NO DATA SAVED!")
        
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
        
        print(f"\nROBUST PROCESSING SETTINGS:")
        print(f"   - Processing batch size: {self.processing_batch_size} chunks")
        print(f"   - Error recovery: Individual chunk processing")
        print(f"   - Binary data detection: Enabled")
        print(f"   - Content cleaning: Enabled")
        
        print("=" * 60)
        
        if success:
            print("SUCCESS: Ready for RAG queries! All documents indexed successfully.")
        elif results['total_saved'] > 0:
            print("WARNING: Ready for RAG queries with partial data.")
            print("INFO: Check error logs for details on failed items:")
            print("   - failed_chunks.log")
            print("   - embedding_errors.log") 
            print("   - batch_failures.log")
            print("   - invalid_chunks_report.log")
        else:
            print("ERROR: No data available for RAG queries.")
        
        return success
    
    def write_comprehensive_log(self, results, deletion_info, encoding_issues=0, error_log_file="./indexing_errors.log"):
        """
        Write comprehensive log of the processing session
        
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
                f.write(f"\n--- Robust processing run at {timestamp} ---\n")
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
                f.write("-------------------------------------\n\n")
        except Exception as e:
            print(f"WARNING: Could not write to {error_log_file}: {e}")


class ProgressTracker:
    """Utility class for tracking and displaying progress"""
    
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
        
        print("\nProgress Summary:")
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
        print("-" * 50)


def create_batch_processor(embedding_processor, processing_batch_size=100):
    """
    Create a batch processor instance
    
    Args:
        embedding_processor: EmbeddingProcessor instance
        processing_batch_size: Number of chunks per batch
    
    Returns:
        BatchProcessor: Configured processor
    """
    return BatchProcessor(embedding_processor, processing_batch_size)


def create_progress_tracker():
    """
    Create a progress tracker instance
    
    Returns:
        ProgressTracker: New progress tracker
    """
    return ProgressTracker()