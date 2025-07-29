#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core file utilities module for RAG Document Indexer
Basic utility functions without circular dependencies
ENHANCED: Added resilient file processing with graceful degradation for missing dependencies
"""

import os
import sys
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


def safe_import_with_fallback(module_name, package=None, fallback_message=None):
    """
    Safely import a module with graceful fallback
    
    Args:
        module_name: Name of the module to import
        package: Package name if doing relative import
        fallback_message: Custom message to show if import fails
    
    Returns:
        tuple: (module_object_or_None, success_boolean, error_message)
    """
    try:
        if package:
            module = __import__(module_name, fromlist=[package])
        else:
            module = __import__(module_name)
        return module, True, None
    except ImportError as e:
        error_msg = fallback_message or f"Optional dependency '{module_name}' not available: {str(e)}"
        return None, False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error importing '{module_name}': {str(e)}"
        return None, False, error_msg


def safe_read_file_with_fallbacks(file_path, max_size=50*1024*1024):
    """
    Safely read file with multiple fallback strategies for different file types
    ENHANCED: Now handles various file formats with graceful degradation
    
    Args:
        file_path: Path to the file to read
        max_size: Maximum file size in bytes (default 50MB)
    
    Returns:
        tuple: (content, error_code, processing_info)
    """
    try:
        # Check file size first
        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            return None, "FILE_TOO_LARGE", {'size': file_size, 'max_allowed': max_size}
        
        if file_size == 0:
            return None, "EMPTY_FILE", {'size': 0}
        
        file_ext = os.path.splitext(file_path)[1].lower()
        processing_info = {'file_extension': file_ext, 'file_size': file_size, 'methods_tried': []}
        
        # Strategy 1: Handle Excel files with multiple fallbacks
        if file_ext in ['.xlsx', '.xls']:
            return _read_excel_file_resilient(file_path, processing_info)
        
        # Strategy 2: Handle CSV files with multiple fallbacks
        elif file_ext == '.csv':
            return _read_csv_file_resilient(file_path, processing_info)
        
        # Strategy 3: Handle other structured data files
        elif file_ext in ['.json', '.xml', '.yaml', '.yml']:
            return _read_structured_file_resilient(file_path, processing_info)
        
        # Strategy 4: Handle text files with encoding detection
        elif file_ext in ['.txt', '.md', '.rst', '.log', '.py', '.js', '.html', '.htm']:
            return _read_text_file_resilient(file_path, processing_info)
        
        # Strategy 5: Handle binary files that might contain text
        elif file_ext in ['.doc', '.rtf']:
            return _read_binary_text_file_resilient(file_path, processing_info)
        
        # Strategy 6: Default text reading with multiple encodings
        else:
            return _read_unknown_file_resilient(file_path, processing_info)
        
    except Exception as e:
        return None, f"FATAL_ERROR_{type(e).__name__}", {'error': str(e)}


def _read_excel_file_resilient(file_path, processing_info):
    """
    Read Excel files with multiple fallback strategies
    
    Args:
        file_path: Path to Excel file
        processing_info: Processing information dict
    
    Returns:
        tuple: (content, error_code, processing_info)
    """
    file_ext = processing_info['file_extension']
    
    # Strategy 1: Try pandas with openpyxl (recommended for .xlsx)
    if file_ext == '.xlsx':
        pandas_module, pandas_success, pandas_error = safe_import_with_fallback('pandas')
        if pandas_success:
            try:
                processing_info['methods_tried'].append('pandas_openpyxl')
                df = pandas_module.read_excel(file_path, engine='openpyxl')
                content = df.to_string()
                content = clean_content_from_null_bytes(content)
                if content.strip():
                    processing_info['success_method'] = 'pandas_openpyxl'
                    return content, None, processing_info
            except ImportError:
                processing_info['methods_tried'].append('pandas_openpyxl_missing_engine')
            except Exception as e:
                processing_info['methods_tried'].append(f'pandas_openpyxl_error_{type(e).__name__}')
    
    # Strategy 2: Try pandas with xlrd (for .xls files or .xlsx fallback)
    pandas_module, pandas_success, pandas_error = safe_import_with_fallback('pandas')
    if pandas_success:
        try:
            processing_info['methods_tried'].append('pandas_xlrd')
            df = pandas_module.read_excel(file_path, engine='xlrd' if file_ext == '.xls' else None)
            content = df.to_string()
            content = clean_content_from_null_bytes(content)
            if content.strip():
                processing_info['success_method'] = 'pandas_xlrd'
                return content, None, processing_info
        except ImportError as e:
            processing_info['methods_tried'].append('pandas_xlrd_missing_dependency')
            processing_info['xlrd_error'] = str(e)
        except Exception as e:
            processing_info['methods_tried'].append(f'pandas_xlrd_error_{type(e).__name__}')
    
    # Strategy 3: Try openpyxl directly (for .xlsx)
    if file_ext == '.xlsx':
        openpyxl_module, openpyxl_success, openpyxl_error = safe_import_with_fallback('openpyxl')
        if openpyxl_success:
            try:
                processing_info['methods_tried'].append('openpyxl_direct')
                workbook = openpyxl_module.load_workbook(file_path, data_only=True)
                content_parts = []
                for sheet_name in workbook.sheetnames:
                    sheet = workbook[sheet_name]
                    for row in sheet.iter_rows(values_only=True):
                        row_text = '\t'.join([str(cell) if cell is not None else '' for cell in row])
                        if row_text.strip():
                            content_parts.append(row_text)
                
                content = '\n'.join(content_parts)
                content = clean_content_from_null_bytes(content)
                if content.strip():
                    processing_info['success_method'] = 'openpyxl_direct'
                    return content, None, processing_info
            except Exception as e:
                processing_info['methods_tried'].append(f'openpyxl_direct_error_{type(e).__name__}')
    
    # Strategy 4: Try xlrd directly (for .xls)
    if file_ext == '.xls':
        xlrd_module, xlrd_success, xlrd_error = safe_import_with_fallback('xlrd')
        if xlrd_success:
            try:
                processing_info['methods_tried'].append('xlrd_direct')
                workbook = xlrd_module.open_workbook(file_path)
                content_parts = []
                for sheet_index in range(workbook.nsheets):
                    sheet = workbook.sheet_by_index(sheet_index)
                    for row_index in range(sheet.nrows):
                        row_values = sheet.row_values(row_index)
                        row_text = '\t'.join([str(cell) for cell in row_values])
                        if row_text.strip():
                            content_parts.append(row_text)
                
                content = '\n'.join(content_parts)
                content = clean_content_from_null_bytes(content)
                if content.strip():
                    processing_info['success_method'] = 'xlrd_direct'
                    return content, None, processing_info
            except Exception as e:
                processing_info['methods_tried'].append(f'xlrd_direct_error_{type(e).__name__}')
    
    # All proper methods failed - SKIP the file (no binary extraction)
    error_details = {
        'pandas_available': pandas_success,
        'pandas_error': pandas_error if not pandas_success else None,
        'methods_attempted': len(processing_info['methods_tried']),
        'file_type': 'Excel',
        'suggestion': 'Install pandas and openpyxl/xlrd: pip install pandas openpyxl xlrd',
        'action_taken': 'File skipped - no suitable reader available'
    }
    processing_info['error_details'] = error_details
    
    return None, "SKIP_EXCEL_FILE_NO_READER", processing_info


def _read_csv_file_resilient(file_path, processing_info):
    """
    Read CSV files with multiple fallback strategies
    
    Args:
        file_path: Path to CSV file
        processing_info: Processing information dict
    
    Returns:
        tuple: (content, error_code, processing_info)
    """
    # Strategy 1: Try pandas
    pandas_module, pandas_success, pandas_error = safe_import_with_fallback('pandas')
    if pandas_success:
        try:
            processing_info['methods_tried'].append('pandas_csv')
            df = pandas_module.read_csv(file_path)
            content = df.to_string()
            content = clean_content_from_null_bytes(content)
            if content.strip():
                processing_info['success_method'] = 'pandas_csv'
                return content, None, processing_info
        except Exception as e:
            processing_info['methods_tried'].append(f'pandas_csv_error_{type(e).__name__}')
    
    # Strategy 2: Try built-in csv module
    try:
        import csv
        processing_info['methods_tried'].append('builtin_csv')
        
        content_parts = []
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                row_text = '\t'.join(row)
                content_parts.append(row_text)
        
        content = '\n'.join(content_parts)
        content = clean_content_from_null_bytes(content)
        if content.strip():
            processing_info['success_method'] = 'builtin_csv'
            return content, None, processing_info
    except Exception as e:
        processing_info['methods_tried'].append(f'builtin_csv_error_{type(e).__name__}')
    
    # Fallback to text reading only (no binary extraction)
    return _read_text_file_resilient(file_path, processing_info)


def _read_structured_file_resilient(file_path, processing_info):
    """
    Read structured data files (JSON, XML, YAML) with fallbacks
    
    Args:
        file_path: Path to structured file
        processing_info: Processing information dict
    
    Returns:
        tuple: (content, error_code, processing_info)
    """
    file_ext = processing_info['file_extension']
    
    # Strategy 1: Try appropriate parser
    if file_ext == '.json':
        try:
            import json
            processing_info['methods_tried'].append('json_parser')
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            content = json.dumps(data, indent=2, ensure_ascii=False)
            content = clean_content_from_null_bytes(content)
            if content.strip():
                processing_info['success_method'] = 'json_parser'
                return content, None, processing_info
        except Exception as e:
            processing_info['methods_tried'].append(f'json_parser_error_{type(e).__name__}')
    
    elif file_ext == '.xml':
        # Try xml.etree.ElementTree
        try:
            import xml.etree.ElementTree as ET
            processing_info['methods_tried'].append('xml_etree')
            tree = ET.parse(file_path)
            root = tree.getroot()
            content = ET.tostring(root, encoding='unicode')
            content = clean_content_from_null_bytes(content)
            if content.strip():
                processing_info['success_method'] = 'xml_etree'
                return content, None, processing_info
        except Exception as e:
            processing_info['methods_tried'].append(f'xml_etree_error_{type(e).__name__}')
    
    elif file_ext in ['.yaml', '.yml']:
        # Try yaml parser
        yaml_module, yaml_success, yaml_error = safe_import_with_fallback('yaml')
        if yaml_success:
            try:
                processing_info['methods_tried'].append('yaml_parser')
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml_module.safe_load(f)
                content = yaml_module.dump(data, default_flow_style=False, allow_unicode=True)
                content = clean_content_from_null_bytes(content)
                if content.strip():
                    processing_info['success_method'] = 'yaml_parser'
                    return content, None, processing_info
            except Exception as e:
                processing_info['methods_tried'].append(f'yaml_parser_error_{type(e).__name__}')
    
    # Fallback to text reading
    return _read_text_file_resilient(file_path, processing_info)


def _read_text_file_resilient(file_path, processing_info):
    """
    Read text files with multiple encoding strategies
    
    Args:
        file_path: Path to text file
        processing_info: Processing information dict
    
    Returns:
        tuple: (content, error_code, processing_info)
    """
    # Define encoding strategies in order of preference
    encoding_strategies = [
        ('utf-8', 'UTF-8 standard'),
        ('utf-8-sig', 'UTF-8 with BOM'),
        ('latin-1', 'Latin-1 fallback'),
        ('cp1252', 'Windows-1252'),
        ('iso-8859-1', 'ISO-8859-1'),
        ('ascii', 'ASCII strict'),
    ]
    
    for encoding, description in encoding_strategies:
        try:
            processing_info['methods_tried'].append(f'encoding_{encoding}')
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            content = clean_content_from_null_bytes(content)
            if content.strip():
                processing_info['success_method'] = f'encoding_{encoding}'
                processing_info['encoding_used'] = encoding
                error_code = None if encoding == 'utf-8' else f"ENCODING_FALLBACK_{encoding.upper()}"
                return content, error_code, processing_info
        except UnicodeDecodeError:
            continue
        except Exception as e:
            processing_info['methods_tried'].append(f'encoding_{encoding}_error_{type(e).__name__}')
    
    # Final fallback: forced decoding with character filtering (limited scope)
    try:
        processing_info['methods_tried'].append('forced_text_decode')
        with open(file_path, 'rb') as f:
            raw_content = f.read()
        
        content = raw_content.decode('utf-8', errors='replace')
        content = content.replace('\ufffd', ' ')  # Remove replacement chars
        content = ''.join(c for c in content if c.isprintable() or c.isspace())
        content = clean_content_from_null_bytes(content)
        
        if content.strip():
            processing_info['success_method'] = 'forced_text_decode'
            processing_info['warning'] = 'Forced text decoding used - some characters may be lost'
            return content, "FORCED_TEXT_DECODE", processing_info
    except Exception as e:
        processing_info['methods_tried'].append(f'forced_text_decode_error_{type(e).__name__}')
    
    # All text methods failed - skip the file
    return None, "SKIP_FILE_NO_TEXT_READER", processing_info


def _read_binary_text_file_resilient(file_path, processing_info):
    """
    Read binary files that might contain text (like .doc, .rtf) - SKIP if proper libraries missing
    
    Args:
        file_path: Path to binary file
        processing_info: Processing information dict
    
    Returns:
        tuple: (content, error_code, processing_info)
    """
    file_ext = processing_info['file_extension']
    
    # Strategy 1: Try specialized libraries ONLY - no binary fallback
    if file_ext == '.doc':
        # Try python-docx2txt
        docx2txt_module, docx2txt_success, docx2txt_error = safe_import_with_fallback('docx2txt')
        if docx2txt_success:
            try:
                processing_info['methods_tried'].append('docx2txt_doc')
                content = docx2txt_module.process(file_path)
                content = clean_content_from_null_bytes(content)
                if content and content.strip():
                    processing_info['success_method'] = 'docx2txt_doc'
                    return content, None, processing_info
            except Exception as e:
                processing_info['methods_tried'].append(f'docx2txt_doc_error_{type(e).__name__}')
    
    elif file_ext == '.rtf':
        # Try striprtf
        striprtf_module, striprtf_success, striprtf_error = safe_import_with_fallback('striprtf.striprtf', 'striprtf')
        if striprtf_success:
            try:
                processing_info['methods_tried'].append('striprtf')
                with open(file_path, 'r', encoding='latin-1') as f:
                    rtf_content = f.read()
                content = striprtf_module.rtf_to_text(rtf_content)
                content = clean_content_from_null_bytes(content)
                if content and content.strip():
                    processing_info['success_method'] = 'striprtf'
                    return content, None, processing_info
            except Exception as e:
                processing_info['methods_tried'].append(f'striprtf_error_{type(e).__name__}')
    
    # No proper library available - SKIP the file (no binary extraction)
    processing_info['error_details'] = {
        'file_type': file_ext.upper(),
        'required_library': 'docx2txt' if file_ext == '.doc' else 'striprtf',
        'suggestion': f'Install required library: pip install {"docx2txt" if file_ext == ".doc" else "striprtf"}',
        'action_taken': 'File skipped - no proper library available'
    }
    
    return None, f"SKIP_{file_ext.upper()}_FILE_NO_LIBRARY", processing_info


def _read_unknown_file_resilient(file_path, processing_info):
    """
    Read files with unknown extensions using text-only strategies
    
    Args:
        file_path: Path to unknown file
        processing_info: Processing information dict
    
    Returns:
        tuple: (content, error_code, processing_info)
    """
    # Try text reading first and only
    result = _read_text_file_resilient(file_path, processing_info)
    if result[0] is not None:  # Success
        return result
    
    # Check if file might be mostly binary (and thus should be skipped)
    try:
        processing_info['methods_tried'].append('binary_detection')
        with open(file_path, 'rb') as f:
            sample = f.read(1000)  # Read first 1000 bytes
        
        # Count null bytes and non-printable characters
        null_bytes = sample.count(b'\x00')
        non_printable = sum(1 for byte in sample if byte < 32 and byte not in [9, 10, 13])  # Tab, LF, CR are OK
        
        # If more than 10% null bytes or 30% non-printable, it's likely binary
        if len(sample) > 0:
            null_ratio = null_bytes / len(sample)
            non_printable_ratio = non_printable / len(sample)
            
            if null_ratio > 0.1 or non_printable_ratio > 0.3:
                processing_info['error_details'] = {
                    'file_type': 'Unknown binary file',
                    'null_byte_ratio': null_ratio,
                    'non_printable_ratio': non_printable_ratio,
                    'action_taken': 'File skipped - appears to be binary data'
                }
                return None, "SKIP_UNKNOWN_BINARY_FILE", processing_info
        
        # If it seems like text but text reading failed, try one more encoding
        try:
            processing_info['methods_tried'].append('latin1_last_attempt')
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            
            content = clean_content_from_null_bytes(content)
            if content.strip():
                processing_info['success_method'] = 'latin1_last_attempt'
                processing_info['warning'] = 'Unknown file type processed with latin-1 encoding'
                return content, "UNKNOWN_FILE_LATIN1", processing_info
        except Exception as e:
            processing_info['methods_tried'].append(f'latin1_last_attempt_error_{type(e).__name__}')
    
    except Exception as e:
        processing_info['methods_tried'].append(f'binary_detection_error_{type(e).__name__}')
    
    # All methods failed - skip the file
    processing_info['error_details'] = {
        'file_type': 'Unknown file type',
        'methods_attempted': len(processing_info['methods_tried']),
        'action_taken': 'File skipped - no suitable text reader found'
    }
    
    return None, "SKIP_UNKNOWN_FILE_NO_READER", processing_info


def safe_read_file(file_path, max_size=50*1024*1024):
    """
    Legacy wrapper for backward compatibility
    
    Args:
        file_path: Path to the file to read
        max_size: Maximum file size in bytes (default 50MB)
    
    Returns:
        tuple: (content, error_code) - simplified for backward compatibility
    """
    content, error_code, processing_info = safe_read_file_with_fallbacks(file_path, max_size)
    return content, error_code


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
            'is_excel': path_obj.suffix.lower() in ['.xlsx', '.xls'],
            'is_csv': path_obj.suffix.lower() == '.csv',
            'supports_advanced_parsing': path_obj.suffix.lower() in ['.docx', '.doc'],
            'requires_special_handling': path_obj.suffix.lower() in ['.xlsx', '.xls', '.csv', '.xml', '.json']
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
        'excel_files': 0,
        'csv_files': 0,
        'other_files': 0,
        'total_size': 0,
        'large_files': [],
        'problematic_files': [],
        'advanced_parsing_candidates': 0,
        'special_handling_files': 0,
        'file_extensions': {},
        'directories_scanned': 0,
        'directories_skipped': 0,
        'blacklisted_directories': [],
        'scan_errors': [],
        'resilient_processing_needed': []
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
        
        # Analyze each file with resilient processing detection
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
                elif file_info['is_excel']:
                    stats['excel_files'] += 1
                    stats['resilient_processing_needed'].append(file_path)
                elif file_info['is_csv']:
                    stats['csv_files'] += 1
                    stats['resilient_processing_needed'].append(file_path)
                else:
                    stats['other_files'] += 1
                
                # Count files that support advanced parsing
                if file_info['supports_advanced_parsing']:
                    stats['advanced_parsing_candidates'] += 1
                
                # Count files that require special handling
                if file_info['requires_special_handling']:
                    stats['special_handling_files'] += 1
                
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
        
        # Calculate resilient processing statistics
        stats['resilient_processing_ratio'] = (
            len(stats['resilient_processing_needed']) / stats['total_files'] * 100 
            if stats['total_files'] > 0 else 0
        )
        
        if verbose:
            print(f"?? Scan complete:")
            print(f"   Files found: {stats['total_files']}")
            print(f"   Directories scanned: {stats['directories_scanned']}")
            print(f"   Directories skipped: {stats['directories_skipped']}")
            print(f"   Files needing resilient processing: {len(stats['resilient_processing_needed'])}")
            if stats['blacklisted_directories']:
                print(f"   Blacklisted dirs found: {', '.join(stats['blacklisted_directories'][:5])}")
        
        return stats
        
    except Exception as e:
        stats['scan_errors'].append(f"Fatal scan error: {str(e)}")
        return stats


def print_directory_scan_summary(stats, show_blacklist_info=True):
    """
    Print summary of directory scan with blacklist information and resilient processing needs
    
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
    print(f"   Excel files: {stats['excel_files']:,}")
    print(f"   CSV files: {stats['csv_files']:,}")
    print(f"   Image files: {stats['image_files']:,}")
    print(f"   Other files: {stats['other_files']:,}")
    
    print(f"\n?? Processing Info:")
    print(f"   Advanced parsing candidates: {stats['advanced_parsing_candidates']:,}")
    print(f"   Special handling required: {stats['special_handling_files']:,}")
    print(f"   Resilient processing needed: {len(stats['resilient_processing_needed']):,} ({stats['resilient_processing_ratio']:.1f}%)")
    print(f"   Total size: {stats['total_size_mb']:.1f} MB")
    
    if stats['large_files']:
        print(f"   Large files (>10MB): {len(stats['large_files'])}")
    
    if stats['problematic_files']:
        print(f"   ?? Problematic files: {len(stats['problematic_files'])}")
    
    if stats['scan_errors']:
        print(f"   ? Scan errors: {len(stats['scan_errors'])}")
    
    # Show resilient processing recommendations
    if len(stats['resilient_processing_needed']) > 0:
        print(f"\n??? Resilient Processing Recommendations:")
        excel_count = stats['excel_files']
        csv_count = stats['csv_files']
        
        if excel_count > 0:
            print(f"   ?? {excel_count} Excel files detected - ensure pandas/openpyxl/xlrd libraries")
        if csv_count > 0:
            print(f"   ?? {csv_count} CSV files detected - pandas recommended for best results")
        
        print(f"   ??? Resilient processing will handle missing dependencies gracefully")
    
    print()


def get_supported_file_extensions():
    """Get list of supported file extensions for processing"""
    return {
        'documents': ['.txt', '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.rtf'],
        'spreadsheets': ['.xlsx', '.xls', '.csv'],
        'text': ['.md', '.rst', '.log', '.json', '.xml', '.html', '.htm'],
        'images': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'],
        'advanced_parsing': ['.docx', '.doc'],
        'resilient_processing': ['.xlsx', '.xls', '.csv', '.xml', '.json', '.yaml', '.yml'],
        'binary_extraction': []  # Removed - no binary extraction used
    }


def is_supported_file(file_path):
    """
    Check if file is supported for processing
    
    Args:
        file_path: Path to file
    
    Returns:
        tuple: (is_supported, file_category, processing_method)
    """
    file_info = get_file_info(file_path)
    if 'error' in file_info:
        return False, 'error', 'none'
    
    supported_extensions = get_supported_file_extensions()
    file_ext = file_info['suffix']
    
    for category, extensions in supported_extensions.items():
        if file_ext in extensions:
            # Determine processing method
            if category == 'resilient_processing':
                processing_method = 'resilient_fallback'
            elif category == 'advanced_parsing':
                processing_method = 'advanced_parsing'
            else:
                processing_method = 'standard'
            
            return True, category, processing_method
    
    return False, 'unsupported', 'none'


def get_missing_dependencies_report():
    """
    Generate a report of missing optional dependencies
    
    Returns:
        dict: Report of missing dependencies and their impact
    """
    dependencies_to_check = [
        ('pandas', 'Excel and CSV processing'),
        ('openpyxl', 'Modern Excel (.xlsx) files'),
        ('xlrd', 'Legacy Excel (.xls) files'),
        ('yaml', 'YAML file processing'),
        ('docx2txt', 'Legacy .doc file text extraction'),
        ('striprtf', 'RTF file processing'),
        ('lxml', 'Advanced XML processing'),
    ]
    
    report = {
        'available': [],
        'missing': [],
        'impact_summary': {},
        'installation_commands': []
    }
    
    for module_name, description in dependencies_to_check:
        module, success, error = safe_import_with_fallback(module_name)
        if success:
            report['available'].append((module_name, description))
        else:
            report['missing'].append((module_name, description, error))
            report['impact_summary'][module_name] = description
    
    # Generate installation commands
    if report['missing']:
        missing_modules = [item[0] for item in report['missing']]
        report['installation_commands'] = [
            f"pip install {' '.join(missing_modules)}",
            "# Or install individually:",
        ] + [f"pip install {module}" for module in missing_modules]
    
    return report


def print_resilient_processing_status():
    """Print the status of resilient processing capabilities"""
    print("\n??? RESILIENT PROCESSING STATUS:")
    print("=" * 50)
    
    report = get_missing_dependencies_report()
    
    if report['available']:
        print("? Available Dependencies:")
        for module_name, description in report['available']:
            print(f"   ? {module_name}: {description}")
    
    if report['missing']:
        print(f"\n?? Missing Optional Dependencies:")
        for module_name, description, error in report['missing']:
            print(f"   ? {module_name}: {description}")
        
        print(f"\n?? Impact:")
        for module_name, impact in report['impact_summary'].items():
            print(f"   • Without {module_name}: {impact} will use fallback methods")
        
        print(f"\n?? Installation Commands:")
        for command in report['installation_commands']:
            print(f"   {command}")
        
        print(f"\n??? Resilient Processing Features:")
        print(f"   • Files will still be processed using fallback methods")
        print(f"   • Processing will continue even if dependencies are missing")
        print(f"   • Detailed error logging for troubleshooting")
        print(f"   • Graceful degradation with warnings")
    else:
        print("? All optional dependencies are available!")
        print("?? Full resilient processing capabilities enabled")
    
    print("=" * 50)