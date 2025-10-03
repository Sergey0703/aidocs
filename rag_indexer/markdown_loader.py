#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown loader module for RAG Document Indexer (Part 2: Chunking & Vectors Only)
Simple markdown file loading from Docling output
PURPOSE: Load markdown files → validate → create LlamaIndex Documents
"""

import os
from pathlib import Path
from datetime import datetime
from llama_index.core import Document


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
    
    # Remove null bytes and control characters
    content = content.replace('\u0000', '').replace('\x00', '')
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
        cleaned = obj.replace('\u0000', '').replace('\x00', '')
        return cleaned[:1000]  # Limit metadata string length
    else:
        return obj


class MarkdownLoader:
    """Simple markdown file loader for preprocessed documents from Docling"""
    
    def __init__(self, input_dir, recursive=True, config=None):
        """
        Initialize markdown loader
        
        Args:
            input_dir: Input directory path (markdown files from Docling)
            recursive: Whether to scan recursively
            config: Configuration object
        """
        self.input_dir = input_dir
        self.recursive = recursive
        self.config = config
        
        # Get blacklist from config
        self.blacklist_directories = []
        if config:
            self.blacklist_directories = config.BLACKLIST_DIRECTORIES
        
        # Statistics
        self.loading_stats = {
            'total_files_found': 0,
            'markdown_files': 0,
            'documents_created': 0,
            'failed_files': 0,
            'failed_files_list': [],
            'total_characters': 0,
            'directories_scanned': 0,
            'directories_skipped': 0,
            'loading_time': 0
        }
    
    def _is_blacklisted_directory(self, directory_path):
        """
        Check if a directory is blacklisted
        
        Args:
            directory_path: Path to check
        
        Returns:
            bool: True if directory should be excluded
        """
        if not self.blacklist_directories:
            return False
        
        path_obj = Path(directory_path)
        path_parts = path_obj.parts
        
        # Check if any part of the path matches blacklisted directories
        for blacklist_dir in self.blacklist_directories:
            if blacklist_dir in path_parts or path_obj.name == blacklist_dir:
                return True
        
        return False
    
    def _scan_markdown_files(self):
        """
        Scan directory for markdown files
        
        Returns:
            list: List of markdown file paths
        """
        markdown_files = []
        
        try:
            if self.recursive:
                # Walk directory tree
                for root, dirs, files in os.walk(self.input_dir):
                    self.loading_stats['directories_scanned'] += 1
                    
                    # Filter out blacklisted directories
                    if self._is_blacklisted_directory(root):
                        self.loading_stats['directories_skipped'] += 1
                        dirs[:] = []  # Don't recurse into this directory
                        continue
                    
                    # Remove blacklisted directories from traversal
                    original_dirs = dirs.copy()
                    dirs[:] = []
                    for dir_name in original_dirs:
                        dir_path = os.path.join(root, dir_name)
                        if not self._is_blacklisted_directory(dir_path):
                            dirs.append(dir_name)
                        else:
                            self.loading_stats['directories_skipped'] += 1
                    
                    # Find markdown files
                    for file in files:
                        if file.lower().endswith('.md'):
                            file_path = os.path.join(root, file)
                            markdown_files.append(file_path)
                            self.loading_stats['total_files_found'] += 1
            else:
                # Non-recursive scan
                directory_path = Path(self.input_dir)
                for file_path in directory_path.glob('*.md'):
                    if file_path.is_file():
                        markdown_files.append(str(file_path))
                        self.loading_stats['total_files_found'] += 1
                
                self.loading_stats['directories_scanned'] = 1
            
            self.loading_stats['markdown_files'] = len(markdown_files)
            
        except Exception as e:
            print(f"❌ ERROR: Failed to scan directory {self.input_dir}: {e}")
        
        return markdown_files
    
    def _read_markdown_file(self, file_path):
        """
        Read a single markdown file
        
        Args:
            file_path: Path to markdown file
        
        Returns:
            tuple: (content, error_message) - content is None if failed
        """
        try:
            # Try UTF-8 first (should be standard for markdown)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Fallback to UTF-8 with error handling
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            
            # Clean content
            content = clean_content_from_null_bytes(content)
            
            if not content or not content.strip():
                return None, "empty_file"
            
            return content, None
            
        except Exception as e:
            return None, f"read_error: {str(e)}"
    
    def _validate_markdown_content(self, content, file_path):
        """
        Validate markdown content quality
        
        Args:
            content: Markdown content
            file_path: Path to file (for error messages)
        
        Returns:
            tuple: (is_valid, error_reason)
        """
        if not content:
            return False, "empty_content"
        
        # Check minimum length
        min_length = self.config.MIN_CHUNK_LENGTH if self.config else 50
        if len(content.strip()) < min_length:
            return False, f"too_short ({len(content)} chars, min: {min_length})"
        
        # Check if it's mostly text (not binary garbage)
        printable_chars = sum(1 for c in content if c.isprintable() or c.isspace())
        total_chars = len(content)
        
        if total_chars > 0:
            text_ratio = printable_chars / total_chars
            if text_ratio < 0.8:  # Less than 80% printable characters
                return False, f"low_text_quality ({text_ratio:.1%} printable)"
        
        # Check if there are actual words
        words = content.split()
        if len(words) < 3:
            return False, f"too_few_words ({len(words)} words)"
        
        return True, None
    
    def _create_document_from_markdown(self, file_path, content):
        """
        Create LlamaIndex Document from markdown file
        
        Args:
            file_path: Path to markdown file
            content: Markdown content
        
        Returns:
            Document: LlamaIndex Document object
        """
        try:
            # Clean file path and name
            clean_file_path = clean_content_from_null_bytes(str(file_path))
            clean_file_name = clean_content_from_null_bytes(os.path.basename(file_path))
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_mtime = os.path.getmtime(file_path)
            
            # Create metadata
            metadata = {
                'file_path': clean_file_path,
                'file_name': clean_file_name,
                'file_type': 'markdown',
                'file_size': file_size,
                'content_length': len(content),
                'word_count': len(content.split()),
                'indexed_at': datetime.now().isoformat(),
                'source_modified': datetime.fromtimestamp(file_mtime).isoformat(),
                'source_system': 'docling',  # Mark as coming from Docling preprocessing
                'processing_stage': 'chunking_and_vectors'  # Part 2 identifier
            }
            
            # Clean metadata
            clean_metadata = clean_metadata_recursive(metadata)
            
            # Create document
            document = Document(
                text=content,
                metadata=clean_metadata
            )
            
            return document
            
        except Exception as e:
            print(f"   ❌ ERROR: Failed to create document from {file_path}: {e}")
            return None
    
    def load_data(self):
        """
        Load all markdown files and create documents
        
        Returns:
            tuple: (documents, loading_stats)
        """
        import time
        
        print(f"📁 Loading markdown files from: {self.input_dir}")
        
        if self.blacklist_directories:
            print(f"🚫 Blacklisted directories: {', '.join(self.blacklist_directories)}")
        
        start_time = time.time()
        
        # Scan for markdown files
        print("🔍 Scanning directory...")
        markdown_files = self._scan_markdown_files()
        
        if not markdown_files:
            print("⚠️ WARNING: No markdown files found!")
            self.loading_stats['loading_time'] = time.time() - start_time
            return [], self.loading_stats
        
        print(f"📄 Found {len(markdown_files)} markdown files")
        
        if self.loading_stats['directories_skipped'] > 0:
            print(f"🚫 Skipped {self.loading_stats['directories_skipped']} blacklisted directories")
        
        # Load documents
        documents = []
        print("📖 Loading markdown content...")
        
        for i, file_path in enumerate(markdown_files, 1):
            file_name = os.path.basename(file_path)
            
            # Progress indicator
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(markdown_files)} files processed...")
            
            # Read markdown file
            content, error = self._read_markdown_file(file_path)
            
            if content is None:
                self.loading_stats['failed_files'] += 1
                self.loading_stats['failed_files_list'].append(f"{file_name} - {error}")
                print(f"   ⚠️ Failed to read: {file_name} ({error})")
                continue
            
            # Validate content
            is_valid, validation_error = self._validate_markdown_content(content, file_path)
            
            if not is_valid:
                self.loading_stats['failed_files'] += 1
                self.loading_stats['failed_files_list'].append(f"{file_name} - {validation_error}")
                print(f"   ⚠️ Invalid content: {file_name} ({validation_error})")
                continue
            
            # Create document
            document = self._create_document_from_markdown(file_path, content)
            
            if document:
                documents.append(document)
                self.loading_stats['documents_created'] += 1
                self.loading_stats['total_characters'] += len(content)
            else:
                self.loading_stats['failed_files'] += 1
                self.loading_stats['failed_files_list'].append(f"{file_name} - document_creation_failed")
        
        self.loading_stats['loading_time'] = time.time() - start_time
        
        # Print summary
        self._print_loading_summary()
        
        return documents, self.loading_stats
    
    def _print_loading_summary(self):
        """Print loading summary"""
        print(f"\n📊 MARKDOWN LOADING SUMMARY:")
        print(f"   ⏱️ Loading time: {self.loading_stats['loading_time']:.2f}s")
        print(f"   📁 Directories scanned: {self.loading_stats['directories_scanned']}")
        
        if self.loading_stats['directories_skipped'] > 0:
            print(f"   🚫 Directories skipped: {self.loading_stats['directories_skipped']}")
        
        print(f"   📄 Markdown files found: {self.loading_stats['markdown_files']}")
        print(f"   ✅ Documents created: {self.loading_stats['documents_created']}")
        print(f"   ❌ Failed files: {self.loading_stats['failed_files']}")
        print(f"   📝 Total characters: {self.loading_stats['total_characters']:,}")
        
        if self.loading_stats['documents_created'] > 0:
            avg_chars = self.loading_stats['total_characters'] / self.loading_stats['documents_created']
            print(f"   📊 Average characters per document: {avg_chars:.0f}")
            
            success_rate = (self.loading_stats['documents_created'] / self.loading_stats['markdown_files'] * 100)
            print(f"   📈 Success rate: {success_rate:.1f}%")
        
        if self.loading_stats['failed_files'] > 0:
            print(f"\n   ⚠️ Failed files details:")
            for i, failed_file in enumerate(self.loading_stats['failed_files_list'][:5], 1):
                print(f"      {i}. {failed_file}")
            
            if len(self.loading_stats['failed_files_list']) > 5:
                print(f"      ... and {len(self.loading_stats['failed_files_list']) - 5} more")
    
    def get_loading_stats(self):
        """
        Get loading statistics
        
        Returns:
            dict: Loading statistics
        """
        return self.loading_stats.copy()


def create_markdown_loader(documents_dir, recursive=True, config=None):
    """
    Create a markdown loader instance
    
    Args:
        documents_dir: Directory containing markdown files
        recursive: Whether to scan recursively
        config: Configuration object
    
    Returns:
        MarkdownLoader: Configured markdown loader
    """
    return MarkdownLoader(
        input_dir=documents_dir,
        recursive=recursive,
        config=config
    )


def validate_markdown_directory(directory_path):
    """
    Validate that a directory exists and contains markdown files
    
    Args:
        directory_path: Path to directory
    
    Returns:
        tuple: (is_valid, error_message, file_count)
    """
    try:
        if not os.path.exists(directory_path):
            return False, f"Directory does not exist: {directory_path}", 0
        
        if not os.path.isdir(directory_path):
            return False, f"Path is not a directory: {directory_path}", 0
        
        # Count markdown files
        markdown_count = 0
        for root, dirs, files in os.walk(directory_path):
            markdown_count += sum(1 for f in files if f.lower().endswith('.md'))
        
        if markdown_count == 0:
            return False, f"No markdown files found in: {directory_path}", 0
        
        return True, None, markdown_count
        
    except Exception as e:
        return False, f"Error validating directory: {e}", 0


def scan_markdown_files(directory_path, recursive=True):
    """
    Quick scan of markdown files in directory
    
    Args:
        directory_path: Directory to scan
        recursive: Whether to scan recursively
    
    Returns:
        dict: Scan results
    """
    results = {
        'total_markdown_files': 0,
        'total_size_bytes': 0,
        'files': []
    }
    
    try:
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if file.lower().endswith('.md'):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        
                        results['total_markdown_files'] += 1
                        results['total_size_bytes'] += file_size
                        results['files'].append({
                            'path': file_path,
                            'name': file,
                            'size': file_size
                        })
        else:
            directory = Path(directory_path)
            for file_path in directory.glob('*.md'):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    
                    results['total_markdown_files'] += 1
                    results['total_size_bytes'] += file_size
                    results['files'].append({
                        'path': str(file_path),
                        'name': file_path.name,
                        'size': file_size
                    })
        
        # Calculate average size
        if results['total_markdown_files'] > 0:
            results['average_size_bytes'] = results['total_size_bytes'] / results['total_markdown_files']
            results['total_size_mb'] = results['total_size_bytes'] / (1024 * 1024)
        
    except Exception as e:
        results['error'] = str(e)
    
    return results


def print_markdown_scan_summary(scan_results):
    """
    Print summary of markdown file scan
    
    Args:
        scan_results: Results from scan_markdown_files
    """
    if 'error' in scan_results:
        print(f"❌ Scan error: {scan_results['error']}")
        return
    
    print(f"\n📊 MARKDOWN FILES SCAN SUMMARY:")
    print(f"   📄 Total markdown files: {scan_results['total_markdown_files']}")
    
    if scan_results['total_markdown_files'] > 0:
        print(f"   💾 Total size: {scan_results['total_size_mb']:.2f} MB")
        print(f"   📊 Average file size: {scan_results['average_size_bytes']/1024:.1f} KB")
        
        if len(scan_results['files']) <= 10:
            print(f"\n   Files:")
            for file_info in scan_results['files']:
                size_kb = file_info['size'] / 1024
                print(f"      - {file_info['name']} ({size_kb:.1f} KB)")
        else:
            print(f"\n   First 10 files:")
            for file_info in scan_results['files'][:10]:
                size_kb = file_info['size'] / 1024
                print(f"      - {file_info['name']} ({size_kb:.1f} KB)")
            print(f"      ... and {len(scan_results['files']) - 10} more files")


if __name__ == "__main__":
    # Test markdown loader when run directly
    import sys
    
    print("📁 Markdown Loader Test")
    print("=" * 60)
    
    # Test directory validation
    test_dir = "./data/markdown" if len(sys.argv) < 2 else sys.argv[1]
    
    is_valid, error, file_count = validate_markdown_directory(test_dir)
    
    if is_valid:
        print(f"✅ Directory is valid: {test_dir}")
        print(f"📄 Found {file_count} markdown files")
        
        # Quick scan
        scan_results = scan_markdown_files(test_dir, recursive=True)
        print_markdown_scan_summary(scan_results)
    else:
        print(f"❌ Directory validation failed: {error}")
    
    print("=" * 60)