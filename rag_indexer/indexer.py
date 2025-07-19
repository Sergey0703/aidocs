#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust RAG Document Indexer with Error Handling
Indexes documents with batch processing and comprehensive error handling
Handles problematic files gracefully without crashing
"""

import os
import logging
import sys
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import psycopg2

# --- TEXT PROCESSING IMPORTS ---
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter

# --- OCR IMPORTS ---
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("WARNING: OCR libraries not installed. Run: pip install pytesseract pillow opencv-python")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_user_confirmation(prompt, default_no=True):
    """Get user confirmation with default option"""
    default_text = "[y/N]" if default_no else "[Y/n]"
    while True:
        response = input(f"{prompt} {default_text}: ").strip().lower()
        if response == '':
            return not default_no
        elif response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no (or press Enter for default)")

def clean_problematic_node(node):
    """Clean problematic metadata and content from a node"""
    try:
        # Create a copy of the node
        cleaned_node = type(node)(
            text=node.text,
            metadata=node.metadata.copy(),
            embedding=node.embedding
        )
        
        # Clean problematic characters from content
        content = cleaned_node.get_content()
        
        # Remove null bytes and other problematic characters
        content = content.replace('\x00', '').replace('\x01', '').replace('\x02', '')
        
        # Remove control characters (except newlines and tabs)
        cleaned_content = ''.join(char for char in content 
                                if ord(char) >= 32 or char in '\n\t\r')
        
        # Limit content length to prevent oversized chunks
        if len(cleaned_content) > 50000:  # 50KB limit
            cleaned_content = cleaned_content[:50000] + "... [TRUNCATED]"
        
        # Update the node's text
        cleaned_node.text = cleaned_content
        cleaned_node.metadata['text'] = cleaned_content
        
        # Clean metadata values
        for key, value in cleaned_node.metadata.items():
            if isinstance(value, str):
                # Remove problematic characters from metadata strings
                cleaned_value = ''.join(char for char in value 
                                      if ord(char) >= 32 or char in '\n\t\r')
                cleaned_node.metadata[key] = cleaned_value[:1000]  # Limit metadata length
        
        # Add warning flag
        cleaned_node.metadata['cleaned'] = True
        cleaned_node.metadata['original_length'] = len(content)
        
        return cleaned_node
        
    except Exception as e:
        print(f"   ?? Error cleaning node: {e}")
        # Return original node if cleaning fails
        return node

def robust_save_to_database(vector_store, nodes_with_embeddings, batch_num, db_batch_size=25):
    """Robust database saving with error handling for problematic chunks"""
    print(f"Saving {len(nodes_with_embeddings)} chunks to database...")
    db_start_time = time.time()
    
    total_saved = 0
    failed_chunks = []
    
    try:
        # Try to save all chunks at once first
        vector_store.add(nodes_with_embeddings, batch_size=db_batch_size)
        total_saved = len(nodes_with_embeddings)
        db_time = time.time() - db_start_time
        print(f"   ? Saved {total_saved} records in {db_time:.2f}s")
        return total_saved, []
        
    except Exception as e:
        print(f"   ?? Batch save failed: {e}")
        print(f"   ?? Trying individual chunk processing...")
        
        # If batch save fails, try saving chunks individually
        for i, node in enumerate(nodes_with_embeddings):
            try:
                # Clean metadata for problematic chunks
                cleaned_node = clean_problematic_node(node)
                vector_store.add([cleaned_node], batch_size=1)
                total_saved += 1
                
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
                
                print(f"   ? Failed to save chunk {i+1}: {file_name}")
                print(f"      Error: {str(chunk_error)[:100]}...")
        
        db_time = time.time() - db_start_time
        
        if total_saved > 0:
            print(f"   ? Saved {total_saved} records individually in {db_time:.2f}s")
        
        if failed_chunks:
            print(f"   ?? Failed to save {len(failed_chunks)} problematic chunks")
            
            # Write failed chunks to separate log
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
        
        return total_saved, failed_chunks

def robust_embedding_generation(embed_model, batch_nodes, batch_num, embedding_batch_size=5):
    """Robust embedding generation with error handling"""
    print(f"Generating embeddings for {len(batch_nodes)} chunks...")
    embedding_start_time = time.time()
    
    nodes_with_embeddings = []
    embedding_errors = []
    
    # Process embeddings in smaller sub-batches with timestamps
    for j in range(0, len(batch_nodes), embedding_batch_size):
        sub_batch = batch_nodes[j:j + embedding_batch_size]
        
        for i, node in enumerate(sub_batch):
            try:
                # Check for problematic content before embedding
                content = node.get_content()
                
                # Skip chunks with only binary data or very short content
                if len(content.strip()) < 10:
                    print(f"   ?? Skipping chunk {j+i+1}: too short ({len(content)} chars)")
                    continue
                    
                # Check for binary data patterns
                binary_ratio = sum(1 for c in content[:500] if ord(c) < 32 and c not in '\n\t\r') / min(len(content), 500)
                if binary_ratio > 0.3:  # More than 30% binary characters
                    print(f"   ?? Skipping chunk {j+i+1}: binary data detected ({binary_ratio:.1%})")
                    continue
                
                # Generate embedding
                embedding = embed_model.get_text_embedding(content)
                node.embedding = embedding
                nodes_with_embeddings.append(node)
                
            except Exception as e:
                file_name = node.metadata.get('file_name', 'Unknown')
                error_info = {
                    'chunk_index': j+i,
                    'file_name': file_name,
                    'error': str(e),
                    'content_preview': content[:100] + "..." if len(content) > 100 else content
                }
                embedding_errors.append(error_info)
                print(f"   ? Embedding error for chunk {j+i+1} from {file_name}: {str(e)[:50]}...")
        
        # Progress update with detailed timestamps
        processed_in_batch = min(j + embedding_batch_size, len(batch_nodes))
        elapsed = time.time() - embedding_start_time
        chunks_per_second = len(nodes_with_embeddings) / elapsed if elapsed > 0 else 0
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
        
        print(f"   Progress: {batch_num} batches ({progress_pct:.1f}%) | "
              f"Processed: {processed_in_batch}/{len(batch_nodes)} chunks | "
              f"Speed: {chunks_per_second:.1f} chunks/sec | "
              f"Elapsed: {format_time(elapsed)} | "
              f"ETA: {format_time(eta_seconds)} | "
              f"Time: {current_time} | "
              f"Finish: {finish_time}")
        
        # Show checkpoint every 20 sub-batches
        if (j // embedding_batch_size + 1) % 20 == 0:
            checkpoint_time = datetime.now().strftime('%H:%M:%S')
            print(f"   ? Checkpoint at {checkpoint_time}: {processed_in_batch}/{len(batch_nodes)} chunks complete")
    
    embedding_time = time.time() - embedding_start_time
    final_speed = len(nodes_with_embeddings) / embedding_time if embedding_time > 0 else 0
    
    print(f"Embedding generation completed in {embedding_time:.2f} seconds")
    print(f"Average speed: {final_speed:.2f} chunks/second")
    
    if embedding_errors:
        print(f"   ?? {len(embedding_errors)} embedding errors")
        
        # Log embedding errors
        with open('./embedding_errors.log', 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n--- Embedding errors in batch {batch_num} at {timestamp} ---\n")
            for error in embedding_errors:
                f.write(f"File: {error['file_name']}\n")
                f.write(f"Error: {error['error']}\n")
                f.write(f"Preview: {error['content_preview']}\n")
                f.write("-" * 40 + "\n")
    
    return nodes_with_embeddings, embedding_errors

# =============================================================================
# OCR FUNCTIONS
# =============================================================================

def preprocess_image_for_ocr(image_path):
    """Image preprocessing for better OCR quality"""
    try:
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        width, height = image.size
        if width < 1000 or height < 1000:
            scale_factor = max(1000/width, 1000/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        image = image.convert('L')
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        return image
        
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def extract_text_from_image(image_path):
    """Extract text from image using OCR"""
    if not OCR_AVAILABLE:
        return ""
    
    try:
        processed_image = preprocess_image_for_ocr(image_path)
        if processed_image is None:
            return ""
        
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabvgdeiozhziyklmnoprstufhcchshshyeuyaABVGDEIOZHZIYKLMNOPRSTUFHCCHSHSHYEUYA.,!?;:()-[]{}/@#$%^&*+=|\\~`"\' \n\t'
        
        text = pytesseract.image_to_string(processed_image, lang='rus+eng', config=custom_config)
        text = text.strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text
        
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return ""

def process_images_in_directory(directory):
    """Process images in directory and extract text"""
    if not OCR_AVAILABLE:
        print("OCR not available. Skipping image processing.")
        return []
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    documents = []
    processed_count = 0
    successful_count = 0
    
    print("Scanning for images...")
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                file_path = os.path.join(root, file)
                processed_count += 1
                
                print(f"Processing image {processed_count}: {file}")
                
                text = extract_text_from_image(file_path)
                
                if text and len(text) > 20:
                    letters = sum(c.isalpha() for c in text)
                    total_chars = len(text.replace(' ', '').replace('\n', ''))
                    
                    if total_chars > 0 and letters / total_chars > 0.3:
                        doc = Document(
                            text=text,
                            metadata={
                                'file_path': file_path,
                                'file_name': file,
                                'file_type': 'image',
                                'ocr_extracted': True,
                                'text_length': len(text),
                                'confidence_score': letters / total_chars
                            }
                        )
                        documents.append(doc)
                        successful_count += 1
                        print(f"  ? Extracted {len(text)} characters from {file}")
                    else:
                        print(f"  ?? Low quality text from {file}")
                else:
                    print(f"  ?? No meaningful text found in {file}")
    
    print(f"\nImage processing complete:")
    print(f"  Images processed: {processed_count}")
    print(f"  Successful extractions: {successful_count}")
    print(f"  Success rate: {successful_count/processed_count*100:.1f}%" if processed_count > 0 else "  No images found")
    
    return documents

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def check_existing_records(connection_string, files_to_process):
    """Check existing records in database"""
    try:
        conn = psycopg2.connect(connection_string)
        cur = conn.cursor()
        
        total_existing = 0
        existing_files = []
        
        for file_identifier in files_to_process:
            cur.execute("""
                SELECT COUNT(*), metadata->>'file_name' 
                FROM vecs.documents 
                WHERE metadata->>'file_path' = %s 
                   OR metadata->>'file_name' = %s
                GROUP BY metadata->>'file_name'
            """, (file_identifier, file_identifier))
            
            results = cur.fetchall()
            for count, filename in results:
                total_existing += count
                existing_files.append(f"{filename} ({count} records)")
        
        cur.close()
        conn.close()
        
        return total_existing, existing_files
        
    except Exception as e:
        print(f"Error checking existing records: {e}")
        return 0, []

def delete_existing_records(connection_string, files_to_process):
    """Delete existing records from database"""
    try:
        conn = psycopg2.connect(connection_string)
        cur = conn.cursor()
        
        deleted_count = 0
        for file_identifier in files_to_process:
            cur.execute("""
                DELETE FROM vecs.documents 
                WHERE metadata->>'file_path' = %s 
                   OR metadata->>'file_name' = %s
            """, (file_identifier, file_identifier))
            deleted_count += cur.rowcount
        
        conn.commit()
        cur.close()
        conn.close()
        
        return deleted_count
        
    except Exception as e:
        print(f"Error deleting existing records: {e}")
        return 0

def safe_deletion_dialog(connection_string, files_to_process):
    """Safe deletion dialog with confirmation"""
    deletion_info = {'files_processed': 0, 'records_deleted': 0}
    
    if not files_to_process:
        print("? No files to process")
        return deletion_info
    
    total_existing, existing_files = check_existing_records(connection_string, files_to_process)
    
    if total_existing == 0:
        print("? No existing records found - proceeding with clean indexing")
        deletion_info = {
            'files_processed': len(files_to_process),
            'records_deleted': 'No existing records'
        }
        return deletion_info
    
    print(f"\n?? EXISTING RECORDS DETECTED")
    print(f"Found {total_existing} existing records for {len(files_to_process)} files")
    
    if len(existing_files) <= 10:
        print("\nExisting files:")
        for file_info in existing_files:
            print(f"  • {file_info}")
    else:
        print(f"\nFirst 10 existing files:")
        for file_info in existing_files[:10]:
            print(f"  • {file_info}")
        print(f"  ... and {len(existing_files) - 10} more files")
    
    print(f"\n?? AVAILABLE OPTIONS:")
    print("1. ??? DELETE existing records and reindex")
    print("2. ? SKIP deletion and add new records") 
    print("3. ? ABORT indexing")
    
    if total_existing > 1000:
        print(f"\n?? WARNING: {total_existing} records is a large number!")
    
    while True:
        choice = input(f"\n?? Choose option (1/2/3) [default: 2 - skip deletion]: ").strip()
        
        if choice == '' or choice == '2':
            print("? Skipping deletion - will add new records alongside existing ones")
            deletion_info = {
                'files_processed': len(files_to_process),
                'records_deleted': 'Skipped by user choice'
            }
            break
            
        elif choice == '1':
            if total_existing > 100:
                confirm = get_user_confirmation(
                    f"?? Really delete {total_existing} records? This cannot be undone!", 
                    default_no=True
                )
                if not confirm:
                    print("? Deletion cancelled")
                    continue
            
            print("??? Proceeding with deletion...")
            deleted_count = delete_existing_records(connection_string, files_to_process)
            
            if deleted_count > 0:
                print(f"? Successfully deleted {deleted_count} existing records")
            else:
                print("?? No records were deleted (possible error)")
            
            deletion_info = {
                'files_processed': len(files_to_process),
                'records_deleted': deleted_count
            }
            break
            
        elif choice == '3':
            print("? Indexing aborted by user")
            sys.exit(0)
            
        else:
            print("? Invalid choice. Please enter 1, 2, or 3")
    
    return deletion_info

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function with robust batch processing and error handling"""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    load_dotenv()
    
    # --- CONFIGURATION ---
    DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./data/634/2025")
    ERROR_LOG_FILE = "./indexing_errors.log"
    TABLE_NAME = os.getenv("TABLE_NAME", "documents")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
    EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "256"))
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "100"))
    ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
    
    # --- BATCH PROCESSING SETTINGS ---
    PROCESSING_BATCH_SIZE = int(os.getenv("PROCESSING_BATCH_SIZE", "100"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
    DB_BATCH_SIZE = int(os.getenv("DB_BATCH_SIZE", "25"))
    
    # --- CONNECTION ---
    connection_string = os.getenv("SUPABASE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("SUPABASE_CONNECTION_STRING not found in .env file!")
    
    print(f"=== ROBUST RAG INDEXER WITH ERROR HANDLING ===")
    print(f"Documents directory: {DOCUMENTS_DIR}")
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print(f"Vector dimension: {EMBED_DIM}")
    print(f"?? Batch processing: {PROCESSING_BATCH_SIZE} chunks per batch")
    print(f"??? Error handling: Enhanced with recovery")
    print(f"OCR enabled: {ENABLE_OCR and OCR_AVAILABLE}")
    if ENABLE_OCR and not OCR_AVAILABLE:
        print("WARNING: OCR requested but libraries not available!")
    print("=" * 60)
    
    # Initialize components
    print("Initializing LlamaIndex components...")
    vector_store = SupabaseVectorStore(
        postgres_connection_string=connection_string,
        collection_name=TABLE_NAME,
        dimension=EMBED_DIM,
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        paragraph_separator="\n\n",
        secondary_chunking_regex="[.!?]\\s+"
    )
    
    # --- STATISTICS ---
    stats = {
        'start_time': time.time(),
        'documents_loaded': 0,
        'images_processed': 0,
        'chunks_created': 0,
        'embeddings_generated': 0,
        'records_saved': 0,
        'batches_processed': 0,
        'failed_batches': 0,
        'failed_chunks': 0,
        'errors': 0
    }
    
    try:
        # --- 1. LOAD DOCUMENTS ---
        print(f"Loading documents from folder: {DOCUMENTS_DIR}")
        
        # Load text documents
        reader = SimpleDirectoryReader(DOCUMENTS_DIR, recursive=True)
        text_documents = reader.load_data(num_workers=4)
        
        # Load images with OCR
        image_documents = []
        if ENABLE_OCR and OCR_AVAILABLE:
            print("\nProcessing images with OCR...")
            image_documents = process_images_in_directory(DOCUMENTS_DIR)
        
        # Combine all documents
        documents = text_documents + image_documents
        stats['documents_loaded'] = len(text_documents)
        stats['images_processed'] = len(image_documents)
        
        load_time = time.time() - stats['start_time']
        print(f"\nSuccessfully loaded {len(documents)} document objects in {load_time:.2f} seconds.")
        print(f"  Text documents: {len(text_documents)}")
        print(f"  Image documents: {len(image_documents)}")
        
        if not documents:
            print("ERROR: No documents found in the specified directory.")
            return
        
        # --- 2. SAFE DELETION DIALOG ---
        print("\n" + "="*60)
        print("??? SAFE DELETION CHECK")
        print("="*60)
        
        files_to_process = set()
        for doc in documents:
            file_path = doc.metadata.get('file_path', '')
            file_name = doc.metadata.get('file_name', '')
            if file_path:
                files_to_process.add(file_path)
            elif file_name:
                files_to_process.add(file_name)
        
        deletion_info = safe_deletion_dialog(connection_string, files_to_process)
        
        # --- 3. FILTER DOCUMENTS ---
        documents_with_content = []
        failed_documents = []
        
        for doc in documents:
            file_name = doc.metadata.get('file_name', 'Unknown File')
            file_path = doc.metadata.get('file_path', file_name)
            text_content = doc.text.strip()
            
            if not text_content:
                failed_documents.append(f"{file_path} - EMPTY (no text extracted)")
            elif len(text_content) < MIN_CHUNK_LENGTH:
                failed_documents.append(f"{file_path} - TOO SHORT ({len(text_content)} chars)")
            else:
                documents_with_content.append(doc)
        
        if failed_documents:
            print(f"Found {len(failed_documents)} problematic documents. Details will be logged.")
        
        if not documents_with_content:
            print("ERROR: No documents with sufficient text content found. Exiting.")
            return
        
        print(f"Processing {len(documents_with_content)} documents with valid content.")
        
        # --- 4. CREATE ALL CHUNKS ---
        print("\nCreating text chunks from all documents...")
        chunk_start_time = time.time()
        
        try:
            all_nodes = node_parser.get_nodes_from_documents(documents_with_content, show_progress=True)
        except Exception as e:
            print(f"ERROR: Failed to parse documents into chunks: {e}")
            return
        
        chunk_time = time.time() - chunk_start_time
        print(f"Document chunking completed in {chunk_time:.2f} seconds")
        
        # Filter and enhance nodes
        valid_nodes = []
        for node in all_nodes:
            content = node.get_content().strip()
            if (content and 
                len(content) >= MIN_CHUNK_LENGTH and 
                len(content.split()) > 5 and
                not content.isdigit()):
                
                # Add metadata
                if 'file_name' not in node.metadata:
                    node.metadata['file_name'] = node.get_metadata_str()
                node.metadata['text'] = content
                node.metadata['indexed_at'] = datetime.now().isoformat()
                
                valid_nodes.append(node)
        
        stats['chunks_created'] = len(valid_nodes)
        print(f"Created {len(valid_nodes)} valid chunks for processing")
        
        if not valid_nodes:
            print("ERROR: No valid text chunks were generated. Exiting.")
            return
        
        # ======================================================================
        # 5. ROBUST BATCH PROCESSING
        # ======================================================================
        
        total_nodes = len(valid_nodes)
        total_saved = 0
        
        print(f"\n?? Starting robust batch processing of {total_nodes} chunks...")
        print(f"?? Processing batch size: {PROCESSING_BATCH_SIZE} chunks")
        print(f"?? Embedding batch size: {EMBEDDING_BATCH_SIZE} chunks")
        print(f"?? Database batch size: {DB_BATCH_SIZE} chunks")
        print(f"??? Error recovery: Enabled")
        print(f"Starting batch processing at {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
        # Process in batches with robust error handling
        for i in range(0, total_nodes, PROCESSING_BATCH_SIZE):
            batch_nodes = valid_nodes[i:i + PROCESSING_BATCH_SIZE]
            batch_num = i // PROCESSING_BATCH_SIZE + 1
            total_batches = (total_nodes + PROCESSING_BATCH_SIZE - 1) // PROCESSING_BATCH_SIZE
            
            print(f"\n?? Processing batch {batch_num}/{total_batches}")
            print(f"   Chunks {i+1}-{min(i+PROCESSING_BATCH_SIZE, total_nodes)} of {total_nodes}")
            print("-" * 40)
            
            batch_start_time = time.time()
            
            try:
                # Generate embeddings with robust error handling
                nodes_with_embeddings, embedding_errors = robust_embedding_generation(
                    embed_model, batch_nodes, batch_num, EMBEDDING_BATCH_SIZE
                )
                
                stats['embeddings_generated'] += len(nodes_with_embeddings)
                
                # Save to database with robust error handling
                if nodes_with_embeddings:
                    batch_saved, failed_chunks = robust_save_to_database(
                        vector_store, nodes_with_embeddings, batch_num, DB_BATCH_SIZE
                    )
                    
                    total_saved += batch_saved
                    stats['records_saved'] = total_saved
                    
                    if failed_chunks:
                        stats['failed_chunks'] += len(failed_chunks)
                        print(f"   ?? Continuing despite {len(failed_chunks)} failed chunks...")
                else:
                    print(f"   ?? No valid embeddings generated for this batch")
                
                batch_time = time.time() - batch_start_time
                stats['batches_processed'] += 1
                
                # Batch summary
                if nodes_with_embeddings:
                    avg_speed = len(nodes_with_embeddings) / (time.time() - batch_start_time)
                    print(f"   ? Batch {batch_num} completed in {batch_time:.2f}s")
                    print(f"   ?? Speed: {avg_speed:.2f} chunks/sec")
                    print(f"   ?? Total saved so far: {total_saved}/{total_nodes}")
                
                # Overall progress estimate
                if batch_num > 1:
                    overall_elapsed = time.time() - stats['start_time']
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
                    print(f"   ?? Overall ETA: {format_time(overall_eta_seconds)} | Overall finish: {overall_finish_time}")
                
            except Exception as e:
                print(f"   ? Batch {batch_num} failed completely: {e}")
                stats['failed_batches'] += 1
                stats['errors'] += 1
                
                # Log the batch failure
                with open('./batch_failures.log', 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n--- Batch {batch_num} failure at {timestamp} ---\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Batch range: {i+1}-{min(i+PROCESSING_BATCH_SIZE, total_nodes)}\n")
                    f.write(f"Files in batch:\n")
                    for node in batch_nodes[:5]:  # Log first 5 files
                        f.write(f"  - {node.metadata.get('file_name', 'Unknown')}\n")
                    f.write("-" * 40 + "\n")
                
                # Continue with next batch
                continue
        
        # ======================================================================
        # END OF BATCH PROCESSING
        # ======================================================================
        
        # Final statistics
        total_time = time.time() - stats['start_time']
        success = stats['errors'] == 0 and stats['failed_chunks'] == 0
        
        print("\n" + "=" * 60)
        if success:
            print("?? ROBUST INDEXING COMPLETED SUCCESSFULLY!")
        elif stats['records_saved'] > 0:
            print("?? ROBUST INDEXING COMPLETED WITH SOME ERRORS!")
            print("? Partial success - some data was saved successfully")
        else:
            print("? INDEXING FAILED - NO DATA SAVED!")
        
        print("=" * 60)
        print(f"?? FINAL STATISTICS:")
        print(f"   Total time: {total_time:.2f}s ({total_time/60:.1f}m)")
        print(f"   Documents processed: {stats['documents_loaded']}")
        print(f"   Images processed: {stats['images_processed']}")
        print(f"   Chunks created: {stats['chunks_created']}")
        print(f"   Embeddings generated: {stats['embeddings_generated']}")
        print(f"   Records saved: {stats['records_saved']}")
        print(f"   Batches processed: {stats['batches_processed']}")
        print(f"   Failed batches: {stats['failed_batches']}")
        print(f"   Failed chunks: {stats['failed_chunks']}")
        
        success_rate = (stats['records_saved'] / stats['chunks_created'] * 100) if stats['chunks_created'] > 0 else 0
        print(f"   Success rate: {success_rate:.1f}%")
        
        if total_time > 0 and stats['embeddings_generated'] > 0:
            print(f"   Average speed: {stats['embeddings_generated']/total_time:.2f} chunks/sec")
        
        print(f"   Records deleted: {deletion_info['records_deleted']}")
        
        print(f"\n?? ROBUST PROCESSING SETTINGS:")
        print(f"   - Processing batch size: {PROCESSING_BATCH_SIZE} chunks")
        print(f"   - Embedding batch size: {EMBEDDING_BATCH_SIZE} chunks")
        print(f"   - Database batch size: {DB_BATCH_SIZE} chunks")
        print(f"   - Error recovery: Individual chunk processing")
        print(f"   - Binary data detection: Enabled")
        print(f"   - Content cleaning: Enabled")
        print(f"   - OCR enabled: {ENABLE_OCR and OCR_AVAILABLE}")
        
        print("=" * 60)
        
        if success:
            print("?? Ready for RAG queries! All documents indexed successfully.")
        elif stats['records_saved'] > 0:
            print("?? Ready for RAG queries with partial data.")
            print("?? Check error logs for details on failed items:")
            print("   - failed_chunks.log")
            print("   - embedding_errors.log") 
            print("   - batch_failures.log")
        else:
            print("? No data available for RAG queries.")
        
        # Write comprehensive log
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n--- Robust processing run at {timestamp} ---\n")
            f.write(f"Status: {'SUCCESS' if success else 'PARTIAL' if stats['records_saved'] > 0 else 'FAILED'}\n")
            f.write(f"Total time: {total_time:.2f}s\n")
            f.write(f"Documents: {stats['documents_loaded']}\n")
            f.write(f"Images: {stats['images_processed']}\n")
            f.write(f"Chunks: {stats['chunks_created']}\n")
            f.write(f"Embeddings: {stats['embeddings_generated']}\n")
            f.write(f"Saved: {stats['records_saved']}\n")
            f.write(f"Failed chunks: {stats['failed_chunks']}\n")
            f.write(f"Failed batches: {stats['failed_batches']}\n")
            f.write(f"Success rate: {success_rate:.1f}%\n")
            f.write(f"Processing batch size: {PROCESSING_BATCH_SIZE}\n")
            if failed_documents:
                f.write("Problematic files:\n")
                for issue in failed_documents:
                    f.write(f"  - {issue}\n")
            f.write("-------------------------------------\n\n")
        
        return success
        
    except KeyboardInterrupt:
        print("\n\n?? Indexing interrupted by user.")
        print(f"?? Partial results: {stats['records_saved']} chunks saved")
        print("??? No data was corrupted - safe to restart.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n?? FATAL ERROR: {e}")
        print("??? Check your configuration and try again.")
        stats['errors'] += 1
        
        # Write error log
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n--- FATAL ERROR at {timestamp} ---\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Partial stats: {stats}\n")
            f.write("-------------------------------------\n\n")
        
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n?? Indexing interrupted by user.")
        print("??? Safe to restart.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n?? FATAL ERROR: {e}")
        sys.exit(1)