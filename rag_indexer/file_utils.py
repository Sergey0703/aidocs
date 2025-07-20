#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File utilities module for RAG Document Indexer
Handles safe file reading with encoding detection, error handling, and basic file scanning
"""

import os
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, Document


def clean_content_from_null_bytes(content):
    """
    Clean content from null bytes and other problematic characters
    
    Args:
        content: Text content to clean
    
    Returns:
        str: Cleaned content
    """
    if not isinstance(content, str):
        return content
    
    # Remove null bytes (\u0000) and other problematic characters - ??????? ???????????!
    content = content.replace('\u0000', '').replace('\x00', '').replace('\x01', '').replace('\x02', '')
    
    # Remove control characters (except newlines and tabs)
    cleaned_content = ''.join(char for char in content 
                            if ord(char) >= 32 or char in '\n\t\r')
    
    return cleaned_content


def safe_read_file(file_path, max_size=50*1024*1024):
    """
    Safely read file with basic error handling for English files
    
    Args:
        file_path: Path to the file to read
        max_size: Maximum file size in bytes (default 50MB)
    
    Returns:
        tuple: (content, error_code) where content is file text or None if failed
    """
    try:
        # Check file size first
        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            return None, "FILE_TOO_LARGE"
        
        if file_size == 0:
            return None, "EMPTY_FILE"
        
        # Try UTF-8 first (standard for English)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # ????: ??????? ?? null bytes!
                content = clean_content_from_null_bytes(content)
                if content.strip():
                    return content, None
        except UnicodeDecodeError:
            # Fallback to Latin-1 for older English files
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    # ????: ??????? ?? null bytes!
                    content = clean_content_from_null_bytes(content)
                    if content.strip():
                        return content, "LATIN1_FALLBACK"
            except Exception:
                pass
        
        # Last resort: binary mode with forced conversion
        try:
            with open(file_path, 'rb') as f:
                raw_content = f.read()
                content = raw_content.decode('utf-8', errors='replace')
                content = content.replace('\ufffd', ' ')  # Remove replacement chars
                content = ''.join(c for c in content if c.isprintable() or c.isspace())
                
                # ????: ??????? ?? null bytes!
                content = clean_content_from_null_bytes(content)
                
                if content.strip():
                    return content, "FORCED_DECODE"
                
        except Exception:
            pass
        
        return None, "NO_READABLE_CONTENT"
        
    except Exception as e:
        return None, f"FATAL_ERROR_{type(e).__name__}"


def scan_files_in_directory(directory, recursive=True):
    """
    Simple scan of directory to get all files
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan recursively
    
    Returns:
        list: List of all file paths in directory
    """
    all_files = []
    
    try:
        if recursive:
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    all_files.append(file_path)
        else:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    all_files.append(item_path)
    except Exception as e:
        print(f"ERROR: Could not scan directory {directory}: {e}")
    
    return all_files


def normalize_file_path(file_path):
    """
    Normalize file path for comparison
    
    Args:
        file_path: File path to normalize
    
    Returns:
        str: Normalized file path
    """
    return os.path.normpath(os.path.abspath(file_path))


def validate_file_path(file_path):
    """
    Validate if file path exists and is readable
    
    Args:
        file_path: Path to validate
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        if not os.path.isfile(file_path):
            return False, "Path is not a file"
        
        if not os.access(file_path, os.R_OK):
            return False, "File is not readable"
        
        return True, None
        
    except Exception as e:
        return False, f"Error accessing file: {e}"


def get_file_info(file_path):
    """
    Get detailed information about a file
    
    Args:
        file_path: Path to the file
    
    Returns:
        dict: File information including size, extension, etc.
    """
    try:
        path_obj = Path(file_path)
        stat_info = os.stat(file_path)
        
        return {
            'name': path_obj.name,
            'stem': path_obj.stem,
            'suffix': path_obj.suffix.lower(),
            'size': stat_info.st_size,
            'size_mb': stat_info.st_size / (1024 * 1024),
            'modified': stat_info.st_mtime,
            'is_text_file': path_obj.suffix.lower() in ['.txt', '.md', '.rst', '.log'],
            'is_document': path_obj.suffix.lower() in ['.pdf', '.docx', '.doc', '.rtf'],
            'is_image': path_obj.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        }
    except Exception as e:
        return {'error': str(e)}


def scan_directory_files(directory, recursive=True):
    """
    Scan directory and return file statistics
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan recursively
    
    Returns:
        dict: Directory statistics
    """
    try:
        stats = {
            'total_files': 0,
            'text_files': 0,
            'document_files': 0,
            'image_files': 0,
            'other_files': 0,
            'total_size': 0,
            'large_files': [],
            'problematic_files': []
        }
        
        if recursive:
            file_iterator = Path(directory).rglob('*')
        else:
            file_iterator = Path(directory).glob('*')
        
        for file_path in file_iterator:
            if file_path.is_file():
                stats['total_files'] += 1
                
                file_info = get_file_info(file_path)
                if 'error' in file_info:
                    stats['problematic_files'].append(str(file_path))
                    continue
                
                stats['total_size'] += file_info['size']
                
                # Categorize files
                if file_info['is_text_file']:
                    stats['text_files'] += 1
                elif file_info['is_document']:
                    stats['document_files'] += 1
                elif file_info['is_image']:
                    stats['image_files'] += 1
                else:
                    stats['other_files'] += 1
                
                # Track large files (>10MB)
                if file_info['size_mb'] > 10:
                    stats['large_files'].append({
                        'path': str(file_path),
                        'size_mb': file_info['size_mb']
                    })
        
        stats['total_size_mb'] = stats['total_size'] / (1024 * 1024)
        return stats
        
    except Exception as e:
        return {'error': str(e)}


class SimpleDirectoryLoader:
    """
    Simple directory loader that uses standard SimpleDirectoryReader
    No complex tracking - just basic loading with standard statistics
    """
    
    def __init__(self, input_dir, recursive=True):
        """Initialize with directory path"""
        self.input_dir = input_dir
        self.recursive = recursive
        self.documents_loaded = 0
        self.loading_time = 0
    
    def load_data(self):
        """
        Load data using standard SimpleDirectoryReader
        
        Returns:
            tuple: (documents, loading_stats, empty_failed_list)
        """
        print("?? Loading documents with SimpleDirectoryReader...")
        
        # Use standard SimpleDirectoryReader
        reader = SimpleDirectoryReader(
            input_dir=self.input_dir,
            recursive=self.recursive
        )
        
        import time
        start_time = time.time()
        
        try:
            documents = reader.load_data()
            self.documents_loaded = len(documents)
            print(f"? Successfully loaded {self.documents_loaded} documents")
        except Exception as e:
            print(f"? Error during document loading: {e}")
            documents = []
            self.documents_loaded = 0
        
        self.loading_time = time.time() - start_time
        
        # Basic statistics (no failed files tracking at this stage)
        loading_stats = {
            'successful_files': self.documents_loaded,  # Approximate
            'failed_files': 0,  # Will be determined later from database
            'encoding_issues': 0,  # Will be determined later
            'total_attempted': 0,  # Will be determined later from directory scan
            'failed_files_detailed': []  # Will be filled later
        }
        
        return documents, loading_stats, []  # Empty failed files list for now
    
    def get_loading_stats(self):
        """
        Get basic loading statistics
        
        Returns:
            dict: Basic loading statistics
        """
        return {
            'successful_files': self.documents_loaded,
            'failed_files': 0,  # To be determined later
            'encoding_issues': 0,  # To be determined later
            'total_attempted': 0,  # To be determined later
            'failed_files_detailed': []
        }
    
    def get_failed_files_list(self):
        """
        Get failed files list (empty for now)
        
        Returns:
            list: Empty list (failed files will be determined later)
        """
        return []
    
    def print_loading_summary(self):
        """Print a basic loading summary"""
        print(f"\n?? Document Loading Summary:")
        print(f"  ?? Documents loaded: {self.documents_loaded}")
        print(f"  ?? Loading time: {self.loading_time:.2f} seconds")
        print(f"  ?? File-level analysis will be performed after database operations")


def create_safe_reader(documents_dir, recursive=True):
    """
    Create a SimpleDirectoryLoader instance
    
    Args:
        documents_dir: Directory to read from
        recursive: Whether to read recursively
    
    Returns:
        SimpleDirectoryLoader: Simple loader instance
    """
    return SimpleDirectoryLoader(
        input_dir=documents_dir,
        recursive=recursive
    )