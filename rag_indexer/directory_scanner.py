#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directory scanner module for RAG Document Indexer
Advanced directory loading with specialized document parsing and hybrid processing
"""

import os
from pathlib import Path
from document_parsers import HybridDocumentProcessor


class AdvancedDirectoryLoader:
    """
    Advanced directory loader with specialized document parsing and hybrid processing
    """
    
    def __init__(self, input_dir, recursive=True, config=None):
        """
        Initialize advanced directory loader
        
        Args:
            input_dir: Input directory path
            recursive: Whether to scan recursively
            config: Configuration object
        """
        self.input_dir = input_dir
        self.recursive = recursive
        self.config = config
        
        # Initialize hybrid processor
        self.hybrid_processor = HybridDocumentProcessor(config)
        
        # Statistics
        self.loading_stats = {
            'total_files_found': 0,
            'docx_files': 0,
            'doc_files': 0,
            'other_files': 0,
            'documents_created': 0,
            'images_extracted': 0,
            'processing_errors': 0,
            'advanced_parsing_used': 0,
            'fallback_used': 0
        }
    
    def set_ocr_processor(self, ocr_processor):
        """
        Set OCR processor for image processing
        
        Args:
            ocr_processor: OCR processor instance
        """
        self.hybrid_processor.set_ocr_processor(ocr_processor)
    
    def scan_files(self):
        """
        Scan directory for files and categorize them
        
        Returns:
            dict: Categorized file lists
        """
        file_categories = {
            'docx_files': [],
            'doc_files': [],
            'other_files': []
        }
        
        try:
            if self.recursive:
                file_iterator = Path(self.input_dir).rglob('*')
            else:
                file_iterator = Path(self.input_dir).glob('*')
            
            for file_path in file_iterator:
                if file_path.is_file():
                    file_ext = file_path.suffix.lower()
                    
                    if file_ext == '.docx':
                        file_categories['docx_files'].append(str(file_path))
                    elif file_ext == '.doc':
                        file_categories['doc_files'].append(str(file_path))
                    else:
                        file_categories['other_files'].append(str(file_path))
                    
                    self.loading_stats['total_files_found'] += 1
            
            self.loading_stats['docx_files'] = len(file_categories['docx_files'])
            self.loading_stats['doc_files'] = len(file_categories['doc_files'])
            self.loading_stats['other_files'] = len(file_categories['other_files'])
            
        except Exception as e:
            print(f"ERROR: Failed to scan directory {self.input_dir}: {e}")
        
        return file_categories
    
    def load_data(self):
        """
        Load and process all documents using advanced parsing
        
        Returns:
            tuple: (documents, loading_stats, failed_files)
        """
        print("Advanced Document Loading with Hybrid Processing")
        
        # Scan files
        file_categories = self.scan_files()
        
        if self.loading_stats['total_files_found'] == 0:
            print("WARNING: No files found in directory")
            return [], self.loading_stats, []
        
        print(f"   Found: {self.loading_stats['docx_files']} .docx, "
              f"{self.loading_stats['doc_files']} .doc, "
              f"{self.loading_stats['other_files']} other files")
        
        all_documents = []
        failed_files = []
        
        # Process DOCX files with advanced parsing
        if file_categories['docx_files']:
            print(f"   Processing {len(file_categories['docx_files'])} DOCX files...")
            for docx_file in file_categories['docx_files']:
                try:
                    documents = self.hybrid_processor.process_docx_file(docx_file)
                    if documents:
                        all_documents.extend(documents)
                        self.loading_stats['documents_created'] += len(documents)
                        self.loading_stats['advanced_parsing_used'] += 1
                        
                        # Count extracted images
                        for doc in documents:
                            if doc.metadata.get('extraction_method') == 'docx_image_ocr':
                                self.loading_stats['images_extracted'] += 1
                    else:
                        failed_files.append(docx_file)
                        self.loading_stats['processing_errors'] += 1
                        
                except Exception as e:
                    print(f"   ERROR: Failed to process {docx_file}: {e}")
                    failed_files.append(docx_file)
                    self.loading_stats['processing_errors'] += 1
        
        # Process DOC files with legacy conversion
        if file_categories['doc_files']:
            print(f"   Processing {len(file_categories['doc_files'])} DOC files...")
            for doc_file in file_categories['doc_files']:
                try:
                    documents = self.hybrid_processor.process_doc_file(doc_file)
                    if documents:
                        all_documents.extend(documents)
                        self.loading_stats['documents_created'] += len(documents)
                        self.loading_stats['advanced_parsing_used'] += 1
                    else:
                        failed_files.append(doc_file)
                        self.loading_stats['processing_errors'] += 1
                        
                except Exception as e:
                    print(f"   ERROR: Failed to process {doc_file}: {e}")
                    failed_files.append(doc_file)
                    self.loading_stats['processing_errors'] += 1
        
        # Process other files with standard processing
        if file_categories['other_files']:
            print(f"   Processing {len(file_categories['other_files'])} other files...")
            
            for other_file in file_categories['other_files']:
                try:
                    # Use simple file reading for non-Word documents
                    documents = self.hybrid_processor._simple_file_processing(other_file)
                    if documents:
                        all_documents.extend(documents)
                        self.loading_stats['documents_created'] += len(documents)
                        self.loading_stats['fallback_used'] += 1
                    else:
                        failed_files.append(other_file)
                        self.loading_stats['processing_errors'] += 1
                        
                except Exception as e:
                    print(f"   ERROR: Failed to process {other_file}: {e}")
                    failed_files.append(other_file)
                    self.loading_stats['processing_errors'] += 1
        
        # Final statistics
        total_success = len(all_documents)
        total_failed = len(failed_files)
        success_rate = (total_success / (total_success + total_failed) * 100) if (total_success + total_failed) > 0 else 0
        
        self.loading_stats.update({
            'total_documents_created': total_success,
            'total_failed': total_failed,
            'success_rate': success_rate
        })
        
        print(f"   SUCCESSFULLY processed: {total_success} documents")
        print(f"   FAILED: {total_failed} files")
        print(f"   Success rate: {success_rate:.1f}%")
        if self.loading_stats['images_extracted'] > 0:
            print(f"   Images extracted and processed: {self.loading_stats['images_extracted']}")
        
        return all_documents, self.loading_stats, failed_files
    
    def get_processing_summary(self):
        """
        Get comprehensive processing summary
        
        Returns:
            dict: Processing summary with detailed statistics
        """
        return {
            'total_files_found': self.loading_stats['total_files_found'],
            'file_breakdown': {
                'docx_files': self.loading_stats['docx_files'],
                'doc_files': self.loading_stats['doc_files'], 
                'other_files': self.loading_stats['other_files']
            },
            'processing_results': {
                'documents_created': self.loading_stats['documents_created'],
                'images_extracted': self.loading_stats['images_extracted'],
                'processing_errors': self.loading_stats['processing_errors'],
                'success_rate': self.loading_stats.get('success_rate', 0)
            },
            'method_usage': {
                'advanced_parsing': self.loading_stats['advanced_parsing_used'],
                'fallback_processing': self.loading_stats['fallback_used']
            },
            'features_enabled': {
                'advanced_parsing': self.hybrid_processor.advanced_parsing_enabled,
                'image_extraction': self.hybrid_processor.extract_images,
                'structure_preservation': self.hybrid_processor.preserve_structure,
                'table_extraction': self.hybrid_processor.extract_tables,
                'hybrid_processing': self.hybrid_processor.hybrid_processing
            }
        }


def scan_directory_files(directory, recursive=True):
    """
    Scan directory and return comprehensive file statistics
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan recursively
    
    Returns:
        dict: Comprehensive directory statistics
    """
    try:
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
            'file_extensions': {}
        }
        
        if recursive:
            file_iterator = Path(directory).rglob('*')
        else:
            file_iterator = Path(directory).glob('*')
        
        for file_path in file_iterator:
            if file_path.is_file():
                stats['total_files'] += 1
                
                # Get file info
                file_ext = file_path.suffix.lower()
                file_size = file_path.stat().st_size
                
                stats['total_size'] += file_size
                
                # Count by extension
                if file_ext in stats['file_extensions']:
                    stats['file_extensions'][file_ext] += 1
                else:
                    stats['file_extensions'][file_ext] = 1
                
                # Categorize files
                if file_ext in ['.txt', '.md', '.rst', '.log']:
                    stats['text_files'] += 1
                elif file_ext in ['.pdf', '.docx', '.doc', '.rtf']:
                    stats['document_files'] += 1
                    if file_ext in ['.docx', '.doc']:
                        stats['word_documents'] += 1
                        if file_ext == '.docx':
                            stats['docx_files'] += 1
                        elif file_ext == '.doc':
                            stats['doc_files'] += 1
                    elif file_ext == '.pdf':
                        stats['pdf_files'] += 1
                elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                    stats['image_files'] += 1
                else:
                    stats['other_files'] += 1
                
                # Count files that support advanced parsing
                if file_ext in ['.docx', '.doc']:
                    stats['advanced_parsing_candidates'] += 1
                
                # Track large files (>10MB)
                if file_size > 10 * 1024 * 1024:
                    stats['large_files'].append({
                        'path': str(file_path),
                        'size_mb': file_size / (1024 * 1024)
                    })
        
        stats['total_size_mb'] = stats['total_size'] / (1024 * 1024)
        return stats
        
    except Exception as e:
        return {'error': str(e)}
