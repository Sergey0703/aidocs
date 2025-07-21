#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding processing module for RAG Document Indexer
Handles embedding generation, node cleaning, and database saving
"""

import time
import os
from datetime import datetime, timedelta


def restart_ollama_if_needed(chunk_index, restart_interval=1000):
    """
    Restart Ollama every N chunks to prevent memory leaks
    
    Args:
        chunk_index: Current chunk number
        restart_interval: Restart every N chunks
    """
    if chunk_index > 0 and chunk_index % restart_interval == 0:
        print(f"\n   INFO: Restarting Ollama after {chunk_index} chunks to prevent memory leaks...")
        try:
            os.system("sudo systemctl restart ollama")
            
            # ?????????? ???????? ?????????? Ollama (?? 30 ??????)
            print(f"   INFO: Waiting for Ollama to initialize...")
            ollama_ready = False
            
            for i in range(30):
                time.sleep(1)
                try:
                    import requests
                    response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if response.status_code == 200:
                        print(f"   SUCCESS: Ollama ready after {i+1} seconds")
                        ollama_ready = True
                        break
                except:
                    continue
            
            if not ollama_ready:
                print(f"   WARNING: Ollama restart took longer than 30 seconds, continuing anyway...")
            
        except Exception as e:
            print(f"   WARNING: Could not restart Ollama: {e}")


def clean_json_recursive(obj):
    """Recursively clean null bytes from all strings in JSON-like structure"""
    if isinstance(obj, dict):
        return {k: clean_json_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_recursive(v) for v in obj]
    elif isinstance(obj, str):
        # Remove null bytes and limit string length
        cleaned = obj.replace('\u0000', '').replace('\x00', '')
        return cleaned[:1000]  # Limit metadata string length
    else:
        return obj


def clean_problematic_node(node):
    """
    Clean problematic metadata and content from a node - ?????????? ???????!
    
    Args:
        node: Node object to clean
    
    Returns:
        Node: Cleaned node object
    """
    try:
        # Create a copy of the node
        cleaned_node = type(node)(
            text=node.text,
            metadata=node.metadata.copy(),
            embedding=node.embedding
        )
        
        # Clean problematic characters from content
        content = cleaned_node.get_content()
        
        # Remove null bytes (\u0000) and other problematic characters - ?????????? ?????!
        content = content.replace('\u0000', '').replace('\x00', '').replace('\x01', '').replace('\x02', '')
        
        # Remove control characters (except newlines and tabs)
        cleaned_content = ''.join(char for char in content 
                                if ord(char) >= 32 or char in '\n\t\r')
        
        # Limit content length to prevent oversized chunks
        if len(cleaned_content) > 50000:  # 50KB limit
            cleaned_content = cleaned_content[:50000] + "... [TRUNCATED]"
        
        # Update the node's text
        cleaned_node.text = cleaned_content
        cleaned_node.metadata['text'] = cleaned_content
        
        # Clean metadata values recursively (??? ??????????!)
        cleaned_node.metadata = clean_json_recursive(cleaned_node.metadata)
        
        # ????: ??????? ?????? ???? LlamaIndex ?? null bytes
        if hasattr(cleaned_node, 'id_') and cleaned_node.id_:
            cleaned_node.id_ = str(cleaned_node.id_).replace('\u0000', '').replace('\x00', '')
        
        if hasattr(cleaned_node, 'doc_id') and cleaned_node.doc_id:
            cleaned_node.doc_id = str(cleaned_node.doc_id).replace('\u0000', '').replace('\x00', '')
        
        # ??????? ref_doc_id ???? ????
        if hasattr(cleaned_node, 'ref_doc_id') and cleaned_node.ref_doc_id:
            cleaned_node.ref_doc_id = str(cleaned_node.ref_doc_id).replace('\u0000', '').replace('\x00', '')
        
        # ??????? source_node ???? ????
        if hasattr(cleaned_node, 'source_node') and cleaned_node.source_node:
            if hasattr(cleaned_node.source_node, 'node_id'):
                cleaned_node.source_node.node_id = str(cleaned_node.source_node.node_id).replace('\u0000', '').replace('\x00', '')
        
        # Add warning flag
        cleaned_node.metadata['cleaned'] = True
        cleaned_node.metadata['original_length'] = len(content)
        
        return cleaned_node
        
    except Exception as e:
        print(f"   WARNING: Error cleaning node: {e}")
        # Return original node if cleaning fails
        return node


def aggressive_clean_all_nodes(nodes):
    """
    ??????????? ??????? ???? nodes ?? null bytes ????? ????????? ? ??
    
    Args:
        nodes: List of nodes to clean
    
    Returns:
        List of cleaned nodes
    """
    cleaned_nodes = []
    
    for node in nodes:
        try:
            # ????????? ??????? ??????? ??????
            cleaned_node = clean_problematic_node(node)
            
            # ?????????????? ??????? - ???????? ??? ????????? ????????
            for attr_name in dir(cleaned_node):
                if not attr_name.startswith('_'):  # Skip private attributes
                    try:
                        attr_value = getattr(cleaned_node, attr_name)
                        if isinstance(attr_value, str):
                            cleaned_value = attr_value.replace('\u0000', '').replace('\x00', '')
                            setattr(cleaned_node, attr_name, cleaned_value)
                    except:
                        pass  # Skip if can't access or modify
            
            cleaned_nodes.append(cleaned_node)
            
        except Exception as e:
            print(f"   WARNING: Failed to clean node completely: {e}")
            # Fallback - try basic cleaning
            try:
                basic_cleaned = clean_problematic_node(node)
                cleaned_nodes.append(basic_cleaned)
            except:
                print(f"   ERROR: Node completely corrupted, skipping...")
                continue
    
    return cleaned_nodes


class EmbeddingProcessor:
    """Processor for generating embeddings and handling database operations"""
    
    def __init__(self, embed_model, vector_store):
        """
        Initialize embedding processor
        
        Args:
            embed_model: Embedding model instance
            vector_store: Vector store instance for database operations
        """
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.stats = {
            'total_processed': 0,
            'successful_embeddings': 0,
            'failed_embeddings': 0,
            'successful_saves': 0,
            'failed_saves': 0
        }
    
    def validate_content_for_embedding(self, content):
        """
        Validate content before embedding generation - ?????????? ?????????
        
        Args:
            content: Text content to validate
        
        Returns:
            tuple: (is_valid, reason)
        """
        # Check minimum length
        if len(content.strip()) < 10:
            return False, f"too_short ({len(content)} chars)"
        
        # ????? ????????: ?????? ?????? ??? PDF/OCR ????????
        # ????????? ???????????? ??????? ??????????? ???????? (??????? ??? ???????? ????????)
        sample = content[:1000]  # ????????? ?????? 1000 ????????
        
        # ??????????? "???????????" ??????? ? ??????
        allowed_special = set('\n\t\r\f\v\x0b\x0c')
        
        # ??????? "??????? ????????" (????????????) ???????
        truly_binary = 0
        for c in sample:
            if ord(c) < 32:  # ??????????? ???????
                if c not in '\n\t\r':  # ????? ?????????
                    truly_binary += 1
            elif ord(c) > 127:  # ??????? UTF-8
                if c not in allowed_special:  # ????? ??????????? ???????????
                    # ?????????, ??? ??????? ????? ????? ??? ??????
                    if not (c.isprintable() or c.isspace() or c.isalnum()):
                        truly_binary += 1
        
        binary_ratio = truly_binary / len(sample) if sample else 0
        
        # ????? ?????: ?????? ???? ????? 90% ???????? ??????!
        if binary_ratio > 0.9:  # ????? ??????? ????? 90%
            return False, f"binary_data_detected ({binary_ratio:.1%})"
        
        # ???????? ?? ??????? ???? - ????? ??????
        letters_digits = sum(1 for c in sample if c.isalnum())
        text_ratio = letters_digits / len(sample) if sample else 0
        
        # ????? ?????: ?????? ???? ????? 10% ????/????!
        if text_ratio < 0.1:  # ????? ?????? ????? 10%
            return False, f"low_text_quality ({text_ratio:.1%})"
        
        # ?????????????? ????????: ???? ?? ???? ?? ??????? ????
        words = content.split()
        if len(words) < 3:
            return False, f"too_few_words ({len(words)} words)"
        
        return True, "valid"
    
    def generate_embedding_for_node(self, node, chunk_index=0):
        """
        Generate embedding for a single node
        
        Args:
            node: Node object to process
            chunk_index: Index of chunk for logging
        
        Returns:
            tuple: (success, error_info)
        """
        try:
            content = node.get_content()
            
            # Validate content
            is_valid, reason = self.validate_content_for_embedding(content)
            if not is_valid:
                return False, f"validation_failed: {reason}"
            
            # Restart Ollama periodically to prevent memory leaks
            restart_ollama_if_needed(chunk_index + 1, restart_interval=400)
            
            # Generate embedding
            embedding = self.embed_model.get_text_embedding(content)
            node.embedding = embedding
            
            self.stats['successful_embeddings'] += 1
            return True, None
            
        except Exception as e:
            error_info = {
                'chunk_index': chunk_index,
                'file_name': node.metadata.get('file_name', 'Unknown'),
                'error': str(e),
                'content_preview': content[:100] + "..." if len(content) > 100 else content
            }
            self.stats['failed_embeddings'] += 1
            return False, error_info
    
    def robust_embedding_generation(self, batch_nodes, batch_num, embedding_batch_size=5):
        """
        Generate embeddings for a batch of nodes with robust error handling
        
        Args:
            batch_nodes: List of nodes to process
            batch_num: Batch number for logging
            embedding_batch_size: Size of sub-batches for processing
        
        Returns:
            tuple: (nodes_with_embeddings, embedding_errors)
        """
        print(f"Generating embeddings for {len(batch_nodes)} chunks...")
        embedding_start_time = time.time()
        
        nodes_with_embeddings = []
        embedding_errors = []
        
        # Process embeddings in smaller sub-batches
        for j in range(0, len(batch_nodes), embedding_batch_size):
            sub_batch = batch_nodes[j:j + embedding_batch_size]
            
            for i, node in enumerate(sub_batch):
                chunk_index = j + i
                absolute_chunk_index = self.stats['total_processed']  # Absolute count for restart
                self.stats['total_processed'] += 1
                
                success, error_info = self.generate_embedding_for_node(node, absolute_chunk_index)
                
                if success:
                    nodes_with_embeddings.append(node)
                else:
                    if isinstance(error_info, dict):
                        embedding_errors.append(error_info)
                        file_name = error_info.get('file_name', 'Unknown')
                        error_msg = error_info.get('error', str(error_info))
                        print(f"   ERROR: Embedding error for chunk {chunk_index+1} from {file_name}: {error_msg[:50]}...")
                    else:
                        print(f"   WARNING: Skipping chunk {chunk_index+1}: {error_info}")
            
            # Progress update with detailed timestamps
            self._print_progress_update(j, batch_nodes, embedding_start_time, batch_num, embedding_batch_size, len(nodes_with_embeddings))
        
        # Final statistics
        embedding_time = time.time() - embedding_start_time
        final_speed = len(nodes_with_embeddings) / embedding_time if embedding_time > 0 else 0
        
        print(f"Embedding generation completed in {embedding_time:.2f} seconds")
        print(f"Average speed: {final_speed:.2f} chunks/second")
        
        if embedding_errors:
            print(f"   WARNING: {len(embedding_errors)} embedding errors")
            self._log_embedding_errors(embedding_errors, batch_num)
        
        return nodes_with_embeddings, embedding_errors
    
    def _print_progress_update(self, j, batch_nodes, start_time, batch_num, embedding_batch_size, successful_count):
        """Print detailed progress update"""
        processed_in_batch = min(j + embedding_batch_size, len(batch_nodes))
        elapsed = time.time() - start_time
        chunks_per_second = successful_count / elapsed if elapsed > 0 else 0
        remaining_chunks = len(batch_nodes) - processed_in_batch
        eta_seconds = remaining_chunks / chunks_per_second if chunks_per_second > 0 else 0
        
        # Format time function
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h {(seconds%3600)/60:.0f}m"
        
        progress_pct = (processed_in_batch / len(batch_nodes)) * 100
        current_time = datetime.now().strftime('%H:%M:%S')
        finish_time = (datetime.now() + timedelta(seconds=eta_seconds)).strftime('%H:%M')
        
        print(f"   Progress: batch {batch_num} ({progress_pct:.1f}%) | "
              f"Processed: {processed_in_batch}/{len(batch_nodes)} chunks | "
              f"Speed: {chunks_per_second:.1f} chunks/sec | "
              f"Elapsed: {format_time(elapsed)} | "
              f"ETA: {format_time(eta_seconds)} | "
              f"Time: {current_time} | "
              f"Finish: {finish_time}")
        
        # Show checkpoint every 20 sub-batches
        if (j // embedding_batch_size + 1) % 20 == 0:
            checkpoint_time = datetime.now().strftime('%H:%M:%S')
            print(f"   CHECKPOINT at {checkpoint_time}: {processed_in_batch}/{len(batch_nodes)} chunks complete")
    
    def _log_embedding_errors(self, embedding_errors, batch_num):
        """Log embedding errors to file"""
        try:
            with open('./embedding_errors.log', 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n--- Embedding errors in batch {batch_num} at {timestamp} ---\n")
                for error in embedding_errors:
                    f.write(f"File: {error.get('file_name', 'Unknown')}\n")
                    f.write(f"Chunk: {error.get('chunk_index', 'Unknown')}\n")
                    f.write(f"Error: {error.get('error', 'Unknown')}\n")
                    f.write(f"Preview: {error.get('content_preview', 'N/A')}\n")
                    f.write("-" * 40 + "\n")
        except Exception as e:
            print(f"   WARNING: Could not write to embedding_errors.log: {e}")
    
    def robust_save_to_database(self, nodes_with_embeddings, batch_num, db_batch_size=25):
        """
        Save nodes to database with robust error handling
        
        Args:
            nodes_with_embeddings: List of nodes with embeddings
            batch_num: Batch number for logging
            db_batch_size: Size of database batches
        
        Returns:
            tuple: (total_saved, failed_chunks)
        """
        print(f"Saving {len(nodes_with_embeddings)} chunks to database...")
        db_start_time = time.time()
        
        total_saved = 0
        failed_chunks = []
        
        # ?????????? ?????: ??????????? ??????? ???? nodes ????? ???????????!
        print(f"   INFO: Cleaning all nodes from null bytes before database save...")
        cleaned_nodes = aggressive_clean_all_nodes(nodes_with_embeddings)
        print(f"   INFO: Cleaned {len(cleaned_nodes)} nodes (original: {len(nodes_with_embeddings)})")
        
        try:
            # Try to save all cleaned chunks at once first
            self.vector_store.add(cleaned_nodes, batch_size=db_batch_size)
            total_saved = len(cleaned_nodes)
            self.stats['successful_saves'] += total_saved
            
            db_time = time.time() - db_start_time
            print(f"   SUCCESS: Saved {total_saved} records in {db_time:.2f}s")
            return total_saved, []
            
        except Exception as e:
            print(f"   WARNING: Batch save failed: {e}")
            print(f"   INFO: Trying individual chunk processing...")
            
            # If batch save fails, try saving chunks individually
            for i, node in enumerate(cleaned_nodes):
                try:
                    # Double-clean problematic chunks
                    ultra_cleaned_node = clean_problematic_node(node)
                    self.vector_store.add([ultra_cleaned_node], batch_size=1)
                    total_saved += 1
                    self.stats['successful_saves'] += 1
                    
                except Exception as chunk_error:
                    # Log the problematic chunk details
                    file_name = node.metadata.get('file_name', 'Unknown')
                    file_path = node.metadata.get('file_path', 'Unknown')
                    chunk_preview = node.get_content()[:100] + "..." if len(node.get_content()) > 100 else node.get_content()
                    
                    failed_info = {
                        'chunk_index': i,
                        'file_name': file_name,
                        'file_path': file_path,
                        'error': str(chunk_error),
                        'content_preview': chunk_preview,
                        'content_length': len(node.get_content())
                    }
                    failed_chunks.append(failed_info)
                    self.stats['failed_saves'] += 1
                    
                    print(f"   ERROR: Failed to save chunk {i+1}: {file_name}")
                    print(f"      Error: {str(chunk_error)[:100]}...")
            
            db_time = time.time() - db_start_time
            
            if total_saved > 0:
                print(f"   SUCCESS: Saved {total_saved} records individually in {db_time:.2f}s")
            
            if failed_chunks:
                print(f"   WARNING: Failed to save {len(failed_chunks)} problematic chunks")
                self._log_failed_chunks(failed_chunks, batch_num)
            
            return total_saved, failed_chunks
    
    def _log_failed_chunks(self, failed_chunks, batch_num):
        """Log failed chunks to file"""
        try:
            with open('./failed_chunks.log', 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n--- Failed chunks in batch {batch_num} at {timestamp} ---\n")
                for failed in failed_chunks:
                    f.write(f"File: {failed['file_name']}\n")
                    f.write(f"Path: {failed['file_path']}\n")
                    f.write(f"Error: {failed['error']}\n")
                    f.write(f"Content length: {failed['content_length']}\n")
                    f.write(f"Preview: {failed['content_preview']}\n")
                    f.write("-" * 40 + "\n")
        except Exception as e:
            print(f"   WARNING: Could not write to failed_chunks.log: {e}")
    
    def get_processing_stats(self):
        """
        Get processing statistics
        
        Returns:
            dict: Processing statistics
        """
        return {
            'total_processed': self.stats['total_processed'],
            'successful_embeddings': self.stats['successful_embeddings'],
            'failed_embeddings': self.stats['failed_embeddings'],
            'successful_saves': self.stats['successful_saves'],
            'failed_saves': self.stats['failed_saves'],
            'embedding_success_rate': (self.stats['successful_embeddings'] / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 0,
            'save_success_rate': (self.stats['successful_saves'] / (self.stats['successful_saves'] + self.stats['failed_saves']) * 100) if (self.stats['successful_saves'] + self.stats['failed_saves']) > 0 else 0
        }
    
    def print_processing_summary(self):
        """Print processing statistics summary"""
        stats = self.get_processing_stats()
        
        print(f"\nEmbedding Processing Summary:")
        print(f"  Total chunks processed: {stats['total_processed']}")
        print(f"  Successful embeddings: {stats['successful_embeddings']}")
        print(f"  Failed embeddings: {stats['failed_embeddings']}")
        print(f"  Embedding success rate: {stats['embedding_success_rate']:.1f}%")
        print(f"  Successful saves: {stats['successful_saves']}")
        print(f"  Failed saves: {stats['failed_saves']}")
        print(f"  Save success rate: {stats['save_success_rate']:.1f}%")
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'total_processed': 0,
            'successful_embeddings': 0,
            'failed_embeddings': 0,
            'successful_saves': 0,
            'failed_saves': 0
        }


class NodeProcessor:
    """Processor for handling node operations and validation"""
    
    def __init__(self, min_chunk_length=100):
        """
        Initialize node processor
        
        Args:
            min_chunk_length: Minimum length for valid chunks
        """
        self.min_chunk_length = min_chunk_length
    
    def validate_node(self, node):
        """
        Validate a node for processing
        
        Args:
            node: Node to validate
        
        Returns:
            tuple: (is_valid, reason)
        """
        content = node.get_content().strip()
        
        if not content:
            return False, "empty_content"
        
        if len(content) < self.min_chunk_length:
            return False, f"too_short ({len(content)} chars)"
        
        if len(content.split()) <= 5:
            return False, "too_few_words"
        
        if content.isdigit():
            return False, "only_digits"
        
        return True, "valid"
    
    def enhance_node_metadata(self, node, indexed_at=None):
        """
        Enhance node metadata with additional information
        
        Args:
            node: Node to enhance
            indexed_at: Timestamp for indexing (defaults to now)
        
        Returns:
            Node: Enhanced node
        """
        if indexed_at is None:
            indexed_at = datetime.now().isoformat()
        
        content = node.get_content()
        
        # Add basic metadata if missing
        if 'file_name' not in node.metadata:
            node.metadata['file_name'] = node.get_metadata_str()
        
        # Add content metadata
        node.metadata.update({
            'text': content,
            'indexed_at': indexed_at,
            'content_length': len(content),
            'word_count': len(content.split()),
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()])
        })
        
        return node
    
    def filter_and_enhance_nodes(self, all_nodes, show_progress=True):
        """
        Filter and enhance a list of nodes
        
        Args:
            all_nodes: List of nodes to process
            show_progress: Whether to show progress updates
        
        Returns:
            tuple: (valid_nodes, invalid_nodes_info)
        """
        valid_nodes = []
        invalid_nodes = []
        
        total_nodes = len(all_nodes)
        indexed_at = datetime.now().isoformat()
        
        # Track invalid files
        invalid_files_summary = {}
        
        for i, node in enumerate(all_nodes):
            if show_progress and i % 1000 == 0:
                print(f"  Processing nodes: {i}/{total_nodes}")
            
            is_valid, reason = self.validate_node(node)
            
            if is_valid:
                enhanced_node = self.enhance_node_metadata(node, indexed_at)
                valid_nodes.append(enhanced_node)
            else:
                file_name = node.metadata.get('file_name', 'Unknown')
                
                invalid_info = {
                    'node_index': i,
                    'reason': reason,
                    'content_preview': node.get_content()[:100],
                    'file_name': file_name,
                    'content_length': len(node.get_content())
                }
                invalid_nodes.append(invalid_info)
                
                # Count invalid reasons per file
                if file_name not in invalid_files_summary:
                    invalid_files_summary[file_name] = {}
                if reason not in invalid_files_summary[file_name]:
                    invalid_files_summary[file_name][reason] = 0
                invalid_files_summary[file_name][reason] += 1
        
        if show_progress:
            print(f"  Node filtering complete: {len(valid_nodes)} valid, {len(invalid_nodes)} invalid")
            
            # Print detailed invalid files report
            if invalid_files_summary:
                print(f"\nInvalid chunks by file:")
                for file_name, reasons in invalid_files_summary.items():
                    total_invalid = sum(reasons.values())
                    reasons_str = ", ".join([f"{reason}: {count}" for reason, count in reasons.items()])
                    print(f"  {file_name}: {total_invalid} invalid chunks ({reasons_str})")
                    
                # Save detailed report to file
                self._save_invalid_chunks_report(invalid_files_summary, invalid_nodes)
        
        return valid_nodes, invalid_nodes
    
    def _save_invalid_chunks_report(self, invalid_files_summary, invalid_nodes):
        """Save detailed report of invalid chunks to file"""
        try:
            report_file = './invalid_chunks_report.log'
            with open(report_file, 'w', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Invalid Chunks Report - {timestamp}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("SUMMARY BY FILE:\n")
                f.write("-" * 30 + "\n")
                for file_name, reasons in invalid_files_summary.items():
                    total_invalid = sum(reasons.values())
                    f.write(f"\nFile: {file_name}\n")
                    f.write(f"Total invalid chunks: {total_invalid}\n")
                    for reason, count in reasons.items():
                        f.write(f"  - {reason}: {count} chunks\n")
                
                f.write(f"\n\nDETAILED INVALID CHUNKS:\n")
                f.write("-" * 30 + "\n")
                for invalid in invalid_nodes[:50]:  # First 50 examples
                    f.write(f"\nFile: {invalid['file_name']}\n")
                    f.write(f"Reason: {invalid['reason']}\n")
                    f.write(f"Content length: {invalid['content_length']}\n")
                    f.write(f"Preview: {invalid['content_preview']}\n")
                    f.write("-" * 20 + "\n")
                
                if len(invalid_nodes) > 50:
                    f.write(f"\n... and {len(invalid_nodes) - 50} more invalid chunks\n")
            
            print(f"Detailed invalid chunks report saved to: {report_file}")
            
        except Exception as e:
            print(f"WARNING: Could not save invalid chunks report: {e}")
    
    def get_node_statistics(self, nodes):
        """
        Get statistics about a list of nodes
        
        Args:
            nodes: List of nodes to analyze
        
        Returns:
            dict: Node statistics
        """
        if not nodes:
            return {'total': 0}
        
        content_lengths = [len(node.get_content()) for node in nodes]
        word_counts = [len(node.get_content().split()) for node in nodes]
        
        # Group by file
        files = {}
        for node in nodes:
            file_name = node.metadata.get('file_name', 'Unknown')
            if file_name not in files:
                files[file_name] = 0
            files[file_name] += 1
        
        return {
            'total': len(nodes),
            'avg_content_length': sum(content_lengths) / len(content_lengths),
            'min_content_length': min(content_lengths),
            'max_content_length': max(content_lengths),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'min_word_count': min(word_counts),
            'max_word_count': max(word_counts),
            'unique_files': len(files),
            'chunks_per_file': sum(files.values()) / len(files)
        }


def create_embedding_processor(embed_model, vector_store):
    """
    Create an embedding processor instance
    
    Args:
        embed_model: Embedding model instance
        vector_store: Vector store instance
    
    Returns:
        EmbeddingProcessor: Configured processor
    """
    return EmbeddingProcessor(embed_model, vector_store)


def create_node_processor(min_chunk_length=100):
    """
    Create a node processor instance
    
    Args:
        min_chunk_length: Minimum length for valid chunks
    
    Returns:
        NodeProcessor: Configured processor
    """
    return NodeProcessor(min_chunk_length)