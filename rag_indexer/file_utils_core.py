#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core file utilities module for RAG Document Indexer
Basic utility functions without circular dependencies
"""

import os
from pathlib import Path


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
    
    # Remove null bytes (\u0000) and other problematic characters
    content = content.replace('\u0000', '').replace('\x00', '').replace('\x01', '').replace('\x02', '')
    
    # Remove control characters (except newlines and tabs)
    cleaned_content = ''.join(char for char in content 
                            if ord(char) >= 32 or char in '\n\t\r')
    
    return cleaned_content


def clean_metadata_recursive(obj):
    """
    Recursively clean metadata from null bytes
    
    Args:
        obj: Object to clean (dict, list, str, etc.)
    
    Returns:
        Cleaned object
    """
    if isinstance(obj, dict):
        return {k: clean_metadata_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_metadata_recursive(v) for v in obj]
    elif isinstance(obj, str):
        # Remove null bytes and limit string length
        cleaned = obj.replace('\u0000', '').replace('\x00', '')
        return cleaned[:1000]  # Limit metadata string length
    else:
        return obj


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
                # Clean null bytes
                content = clean_content_from_null_bytes(content)
                if content.strip():
                    return content, None
        except UnicodeDecodeError:
            # Fallback to Latin-1 for older English files
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    # Clean null bytes
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
                
                # Clean null bytes
                content = clean_content_from_null_bytes(content)
                
                if content.strip():
                    return content, "FORCED_DECODE"
                
        except Exception:
            pass
        
        return None, "NO_READABLE_CONTENT"
        
    except Exception as e:
        return None, f"FATAL_ERROR_{type(e).__name__}"


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
            'is_word_document': path_obj.suffix.lower() in ['.docx', '.doc'],
            'is_image': path_obj.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'],
            'supports_advanced_parsing': path_obj.suffix.lower() in ['.docx', '.doc']
        }
    except Exception as e:
        return {'error': str(e)}


def scan_files_in_directory(directory, recursive=True):
    """
    Scan directory to get all files with detailed categorization
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan recursively
    
    Returns:
        list: List of file paths
    """
    file_list = []
    
    try:
        if recursive:
            file_iterator = Path(directory).rglob('*')
        else:
            file_iterator = Path(directory).glob('*')
        
        for file_path in file_iterator:
            if file_path.is_file():
                file_list.append(str(file_path))
    
    except Exception as e:
        print(f"ERROR: Failed to scan directory {directory}: {e}")
    
    return file_list
