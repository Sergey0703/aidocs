#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File utilities module for RAG Document Indexer
Handles safe file reading with encoding detection, error handling, and failed files tracking
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
    
    # Remove null bytes (\u0000) and other problematic characters - ?????????? ?????!
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
            print(f"   WARNING: Skipping large file {file_path} ({file_size/1024/1024:.1f}MB)")
            return None, "FILE_TOO_LARGE"
        
        if file_size == 0:
            print(f"   WARNING: Skipping empty file {file_path}")
            return None, "EMPTY_FILE"
        
        # Try UTF-8 first (standard for English)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # ?????: ??????? ?? null bytes!
                content = clean_content_from_null_bytes(content)
                if content.strip():
                    print(f"   SUCCESS: Read {file_path} with UTF-8")
                    return content, None
        except UnicodeDecodeError:
            # Fallback to Latin-1 for older English files
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    # ?????: ??????? ?? null bytes!
                    content = clean_content_from_null_bytes(content)
                    if content.strip():
                        print(f"   SUCCESS: Read {file_path} with Latin-1")
                        return content, "LATIN1_FALLBACK"
            except Exception as e:
                print(f"   WARNING: Error reading {file_path} with Latin-1: {e}")
        
        # Last resort: binary mode with forced conversion
        try:
            with open(file_path, 'rb') as f:
                raw_content = f.read()
                content = raw_content.decode('utf-8', errors='replace')
                content = content.replace('\ufffd', ' ')  # Remove replacement chars
                content = ''.join(c for c in content if c.isprintable() or c.isspace())
                
                # ?????: ??????? ?? null bytes!
                content = clean_content_from_null_bytes(content)
                
                if content.strip():
                    print(f"   WARNING: Forcefully read {file_path} with character replacement")
                    return content, "FORCED_DECODE"
                
        except Exception as e:
            print(f"   ERROR: Complete failure reading {file_path}: {e}")
            return None, f"READ_ERROR_{type(e).__name__}"
        
        return None, "NO_READABLE_CONTENT"
        
    except Exception as e:
        print(f"   ERROR: Fatal error accessing {file_path}: {e}")
        return None, f"FATAL_ERROR_{type(e).__name__}"


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


class SafeDirectoryReader(SimpleDirectoryReader):
    """
    Enhanced SimpleDirectoryReader with safe file reading and failed files tracking
    Handles encoding issues gracefully and tracks all failed files with detailed reasons
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with enhanced error tracking"""
        super().__init__(*args, **kwargs)
        self.failed_files = []
        self.failed_files_detailed = []  # NEW: Detailed failed files list
        self.encoding_issues = []
        self.successful_files = []
    
    def load_file(self, input_file, metadata=None, extra_info=None):
        """
        Override load_file to handle encoding issues safely and track failed files
        
        Args:
            input_file: Path to file to load
            metadata: Optional metadata dict
            extra_info: Optional extra info dict
        
        Returns:
            list: List of Document objects (empty if failed)
        """
        file_path_str = str(input_file)
        file_name = os.path.basename(file_path_str)
        
        try:
            # Validate file first
            is_valid, error_msg = validate_file_path(input_file)
            if not is_valid:
                failed_detail = f"{file_name} - VALIDATION_ERROR: {error_msg}"
                self.failed_files.append((input_file, error_msg))
                self.failed_files_detailed.append(failed_detail)
                return []
            
            # Use our safe file reading function
            content, error = safe_read_file(input_file)
            
            if content is None:
                # Determine detailed error reason
                if error == "FILE_TOO_LARGE":
                    failed_detail = f"{file_name} - FILE_TOO_LARGE (>50MB)"
                elif error == "EMPTY_FILE":
                    failed_detail = f"{file_name} - EMPTY_FILE (0 bytes)"
                elif error == "NO_READABLE_CONTENT":
                    failed_detail = f"{file_name} - NO_READABLE_CONTENT"
                elif error.startswith("READ_ERROR"):
                    failed_detail = f"{file_name} - {error}"
                elif error.startswith("FATAL_ERROR"):
                    failed_detail = f"{file_name} - {error}"
                else:
                    failed_detail = f"{file_name} - UNKNOWN_ERROR: {error}"
                
                self.failed_files.append((input_file, error))
                self.failed_files_detailed.append(failed_detail)
                return []
            
            # Check if content is meaningful
            content_length = len(content.strip())
            if content_length == 0:
                failed_detail = f"{file_name} - EMPTY_CONTENT (no text after cleanup)"
                self.failed_files.append((input_file, "EMPTY_CONTENT"))
                self.failed_files_detailed.append(failed_detail)
                return []
            elif content_length < 10:
                failed_detail = f"{file_name} - TOO_SHORT ({content_length} chars)"
                self.failed_files.append((input_file, "TOO_SHORT"))
                self.failed_files_detailed.append(failed_detail)
                return []
            
            # Create metadata
            if metadata is None:
                metadata = {}
            
            # Add encoding info if file had issues
            if error in ["LATIN1_FALLBACK", "FORCED_DECODE"]:
                metadata['encoding_warning'] = True
                metadata['encoding_method'] = error.lower()
                encoding_detail = f"{file_name} - ENCODING_ISSUE: {error}"
                self.encoding_issues.append((input_file, error))
            
            # Clean file paths from null bytes
            clean_file_path = clean_content_from_null_bytes(str(input_file))
            clean_file_name = clean_content_from_null_bytes(os.path.basename(str(input_file)))
            
            # Add file information
            file_info = get_file_info(input_file)
            metadata.update({
                'file_path': clean_file_path,
                'file_name': clean_file_name,
                'file_size': file_info.get('size', 0),
                'file_type': 'text',
                'content_length': content_length
            })
            
            # Clean metadata recursively
            cleaned_metadata = self._clean_metadata_recursive(metadata)
            
            # Create document with cleaned content and metadata
            doc = Document(text=content, metadata=cleaned_metadata)
            self.successful_files.append(str(input_file))
            return [doc]
            
        except Exception as e:
            error_detail = f"{file_name} - LOAD_EXCEPTION: {type(e).__name__}: {str(e)}"
            print(f"   ERROR: Failed to load {input_file}: {e}")
            self.failed_files.append((input_file, f"LOAD_ERROR_{type(e).__name__}"))
            self.failed_files_detailed.append(error_detail)
            return []
    
    def _clean_metadata_recursive(self, obj):
        """
        Recursively clean metadata from null bytes
        
        Args:
            obj: Object to clean (dict, list, str, etc.)
        
        Returns:
            Cleaned object
        """
        if isinstance(obj, dict):
            return {k: self._clean_metadata_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_metadata_recursive(v) for v in obj]
        elif isinstance(obj, str):
            # Remove null bytes and limit string length
            cleaned = obj.replace('\u0000', '').replace('\x00', '')
            return cleaned[:1000]  # Limit metadata string length
        else:
            return obj
    
    def get_loading_stats(self):
        """
        Get statistics about the loading process
        
        Returns:
            dict: Loading statistics with detailed failed files
        """
        return {
            'successful_files': len(self.successful_files),
            'failed_files': len(self.failed_files),
            'encoding_issues': len(self.encoding_issues),
            'total_attempted': len(self.successful_files) + len(self.failed_files),
            'failed_files_detailed': self.failed_files_detailed.copy()  # NEW: Detailed list
        }
    
    def get_failed_files_list(self):
        """
        Get detailed list of failed files for logging
        
        Returns:
            list: List of failed files with detailed reasons
        """
        return self.failed_files_detailed.copy()
    
    def print_loading_summary(self):
        """Print a summary of the loading process"""
        stats = self.get_loading_stats()
        
        print(f"\nFile Loading Summary:")
        print(f"  Successful: {stats['successful_files']}")
        print(f"  Failed: {stats['failed_files']}")
        print(f"  Encoding issues: {stats['encoding_issues']}")
        print(f"  Total attempted: {stats['total_attempted']}")
        
        if self.failed_files:
            print(f"\nFirst 5 failed files:")
            for i, (file_path, error) in enumerate(self.failed_files[:5]):
                print(f"  - {os.path.basename(file_path)}: {error}")
            if len(self.failed_files) > 5:
                print(f"  ... and {len(self.failed_files) - 5} more")
        
        if self.encoding_issues:
            print(f"\nFiles with encoding issues:")
            for file_path, error in self.encoding_issues[:3]:
                print(f"  - {os.path.basename(file_path)}: {error}")
            if len(self.encoding_issues) > 3:
                print(f"  ... and {len(self.encoding_issues) - 3} more")
        
        # Save failed files to log if any
        if self.failed_files_detailed:
            print(f"\n?? Detailed failed files list will be saved to logs")


def create_safe_reader(documents_dir, recursive=True):
    """
    Create a SafeDirectoryReader instance with common settings
    
    Args:
        documents_dir: Directory to read from
        recursive: Whether to read recursively
    
    Returns:
        SafeDirectoryReader: Configured reader instance
    """
    return SafeDirectoryReader(
        input_dir=documents_dir,
        recursive=recursive
    )