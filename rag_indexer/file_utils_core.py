#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core file utilities module for RAG Document Indexer
Basic utility functions without circular dependencies
NEW: Added blacklist directory filtering
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


def is_blacklisted_directory(directory_path, blacklist_directories):
    """
    Check if a directory is in the blacklist and should be excluded
    
    Args:
        directory_path: Path to check
        blacklist_directories: List of blacklisted directory names
    
    Returns:
        bool: True if directory should be excluded
    """
    if not blacklist_directories:
        return False
    
    # Convert path to Path object for easier manipulation
    path_obj = Path(directory_path)
    
    # Get all parts of the path
    path_parts = path_obj.parts
    
    # Check if any part of the path matches blacklisted directories
    for blacklist_dir in blacklist_directories:
        if blacklist_dir in path_parts:
            return True
    
    # Also check the directory name itself
    if path_obj.name in blacklist_directories:
        return True
    
    return False


def should_skip_directory(directory_path, blacklist_directories=None, verbose=False):
    """
    Determine if a directory should be skipped during scanning
    
    Args:
        directory_path: Path to the directory
        blacklist_directories: List of blacklisted directory names
        verbose: Whether to print skip reasons
    
    Returns:
        tuple: (should_skip, reason)
    """
    path_obj = Path(directory_path)
    
    # Check if directory exists and is accessible
    if not path_obj.exists():
        return True, "Directory does not exist"
    
    if not path_obj.is_dir():
        return True, "Path is not a directory"
    
    if not os.access(path_obj, os.R_OK):
        return True, "Directory not readable"
    
    # Check blacklist
    if blacklist_directories and is_blacklisted_directory(directory_path, blacklist_directories):
        if verbose:
            print(f"   ?? Skipping blacklisted directory: {path_obj.name}")
        return True, f"Directory '{path_obj.name}' is blacklisted"
    
    return False, None


def scan_files_in_directory(directory, recursive=True, blacklist_directories=None, verbose=False):
    """
    Scan directory to get all files with blacklist filtering
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan recursively
        blacklist_directories: List of directory names to exclude
        verbose: Whether to print detailed scanning info
    
    Returns:
        list: List of file paths (excludes blacklisted directories)
    """
    file_list = []
    skipped_dirs = []
    
    try:
        if verbose:
            print(f"?? Scanning directory: {directory}")
            if blacklist_directories:
                print(f"?? Blacklisted directories: {', '.join(blacklist_directories)}")
        
        if recursive:
            # Use os.walk for better control over directory traversal
            for root, dirs, files in os.walk(directory):
                # Filter out blacklisted directories from dirs list
                # This prevents os.walk from entering them
                original_dirs = dirs.copy()
                dirs[:] = []  # Clear the list
                
                for dir_name in original_dirs:
                    dir_path = os.path.join(root, dir_name)
                    should_skip, reason = should_skip_directory(dir_path, blacklist_directories, verbose)
                    
                    if should_skip:
                        skipped_dirs.append((dir_path, reason))
                    else:
                        dirs.append(dir_name)  # Add back to dirs for traversal
                
                # Add files from current directory
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    file_list.append(file_path)
        else:
            # Non-recursive scan
            directory_path = Path(directory)
            for item in directory_path.iterdir():
                if item.is_file():
                    file_list.append(str(item))
        
        if verbose and skipped_dirs:
            print(f"?? Skipped {len(skipped_dirs)} blacklisted directories:")
            for skipped_dir, reason in skipped_dirs[:5]:  # Show first 5
                print(f"   - {Path(skipped_dir).name}: {reason}")
            if len(skipped_dirs) > 5:
                print(f"   ... and {len(skipped_dirs) - 5} more")
        
        if verbose:
            print(f"?? Found {len(file_list)} files total")
    
    except Exception as e:
        print(f"? ERROR: Failed to scan directory {directory}: {e}")
    
    return file_list


def scan_directory_with_stats(directory, recursive=True, blacklist_directories=None, verbose=False):
    """
    Scan directory and return comprehensive statistics with blacklist info
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan recursively
        blacklist_directories: List of directory names to exclude
        verbose: Whether to print detailed info
    
    Returns:
        dict: Comprehensive directory statistics including blacklist info
    """
    stats = {
        'total_files': 0,
        'text_files': 0,
        'document_files': 0,
        'word_documents': 0,
        'docx_files': 0,
        'doc_files': 0,
        'image_files': 0,
        'pdf_files': 0,
        'other_files': 0,
        'total_size': 0,
        'large_files': [],
        'problematic_files': [],
        'advanced_parsing_candidates': 0,
        'file_extensions': {},
        'directories_scanned': 0,
        'directories_skipped': 0,
        'blacklisted_directories': [],
        'scan_errors': []
    }
    
    try:
        if verbose:
            print(f"?? Analyzing directory: {directory}")
        
        # Get all files with blacklist filtering
        all_files = scan_files_in_directory(directory, recursive, blacklist_directories, verbose=False)
        
        # Count directories
        if recursive:
            for root, dirs, files in os.walk(directory):
                stats['directories_scanned'] += 1
                
                # Check for blacklisted directories
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if blacklist_directories and is_blacklisted_directory(dir_path, blacklist_directories):
                        stats['directories_skipped'] += 1
                        stats['blacklisted_directories'].append(dir_name)
        
        # Analyze each file
        for file_path in all_files:
            try:
                file_info = get_file_info(file_path)
                
                if 'error' in file_info:
                    stats['problematic_files'].append(file_path)
                    stats['scan_errors'].append(f"{file_path}: {file_info['error']}")
                    continue
                
                stats['total_files'] += 1
                stats['total_size'] += file_info['size']
                
                # Count by extension
                ext = file_info['suffix']
                stats['file_extensions'][ext] = stats['file_extensions'].get(ext, 0) + 1
                
                # Categorize files
                if file_info['is_text_file']:
                    stats['text_files'] += 1
                elif file_info['is_document']:
                    stats['document_files'] += 1
                    if file_info['is_word_document']:
                        stats['word_documents'] += 1
                        if ext == '.docx':
                            stats['docx_files'] += 1
                        elif ext == '.doc':
                            stats['doc_files'] += 1
                    elif ext == '.pdf':
                        stats['pdf_files'] += 1
                elif file_info['is_image']:
                    stats['image_files'] += 1
                else:
                    stats['other_files'] += 1
                
                # Count files that support advanced parsing
                if file_info['supports_advanced_parsing']:
                    stats['advanced_parsing_candidates'] += 1
                
                # Track large files (>10MB)
                if file_info['size_mb'] > 10:
                    stats['large_files'].append({
                        'path': file_path,
                        'size_mb': file_info['size_mb']
                    })
            
            except Exception as e:
                stats['problematic_files'].append(file_path)
                stats['scan_errors'].append(f"{file_path}: {str(e)}")
        
        stats['total_size_mb'] = stats['total_size'] / (1024 * 1024)
        
        # Remove duplicates from blacklisted directories
        stats['blacklisted_directories'] = list(set(stats['blacklisted_directories']))
        
        if verbose:
            print(f"?? Scan complete:")
            print(f"   Files found: {stats['total_files']}")
            print(f"   Directories scanned: {stats['directories_scanned']}")
            print(f"   Directories skipped: {stats['directories_skipped']}")
            if stats['blacklisted_directories']:
                print(f"   Blacklisted dirs found: {', '.join(stats['blacklisted_directories'][:5])}")
        
        return stats
        
    except Exception as e:
        stats['scan_errors'].append(f"Fatal scan error: {str(e)}")
        return stats


def print_directory_scan_summary(stats, show_blacklist_info=True):
    """
    Print summary of directory scan with blacklist information
    
    Args:
        stats: Directory scan statistics
        show_blacklist_info: Whether to show blacklist details
    """
    print(f"\n?? DIRECTORY SCAN SUMMARY:")
    print(f"?? Total files found: {stats['total_files']:,}")
    print(f"?? Directories scanned: {stats['directories_scanned']:,}")
    
    if show_blacklist_info and stats['directories_skipped'] > 0:
        print(f"?? Directories skipped (blacklisted): {stats['directories_skipped']:,}")
        if stats['blacklisted_directories']:
            print(f"   Blacklisted dirs found: {', '.join(stats['blacklisted_directories'][:5])}")
            if len(stats['blacklisted_directories']) > 5:
                print(f"   ... and {len(stats['blacklisted_directories']) - 5} more")
    
    print(f"\n?? File Types:")
    print(f"   Text files: {stats['text_files']:,}")
    print(f"   Document files: {stats['document_files']:,}")
    print(f"     - Word documents: {stats['word_documents']:,} (.docx: {stats['docx_files']}, .doc: {stats['doc_files']})")
    print(f"     - PDF files: {stats['pdf_files']:,}")
    print(f"   Image files: {stats['image_files']:,}")
    print(f"   Other files: {stats['other_files']:,}")
    
    print(f"\n?? Processing Info:")
    print(f"   Advanced parsing candidates: {stats['advanced_parsing_candidates']:,}")
    print(f"   Total size: {stats['total_size_mb']:.1f} MB")
    
    if stats['large_files']:
        print(f"   Large files (>10MB): {len(stats['large_files'])}")
    
    if stats['problematic_files']:
        print(f"   ?? Problematic files: {len(stats['problematic_files'])}")
    
    if stats['scan_errors']:
        print(f"   ? Scan errors: {len(stats['scan_errors'])}")
    
    print()


def get_supported_file_extensions():
    """Get list of supported file extensions for processing"""
    return {
        'documents': ['.txt', '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.rtf'],
        'spreadsheets': ['.xlsx', '.xls', '.csv'],
        'text': ['.md', '.rst', '.log', '.json', '.xml', '.html', '.htm'],
        'images': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'],
        'advanced_parsing': ['.docx', '.doc']
    }


def is_supported_file(file_path):
    """
    Check if file is supported for processing
    
    Args:
        file_path: Path to file
    
    Returns:
        tuple: (is_supported, file_category)
    """
    file_info = get_file_info(file_path)
    if 'error' in file_info:
        return False, 'error'
    
    supported_extensions = get_supported_file_extensions()
    file_ext = file_info['suffix']
    
    for category, extensions in supported_extensions.items():
        if file_ext in extensions:
            return True, category
    
    return False, 'unsupported'