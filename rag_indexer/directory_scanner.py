#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directory scanner module for RAG Document Indexer
Advanced directory loading with specialized document parsing, hybrid processing, and ENHANCED PDF SUPPORT
"""

import os
from pathlib import Path
from document_parsers import HybridDocumentProcessor


class AdvancedDirectoryLoader:
    """
    Advanced directory loader with specialized document parsing, hybrid processing, and PDF support
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
        
        # Initialize hybrid processor with PDF support
        self.hybrid_processor = HybridDocumentProcessor(config)
        
        # Statistics
        self.loading_stats = {
            'total_files_found': 0,
            'docx_files': 0,
            'doc_files': 0,
            'pdf_files': 0,  # NEW: PDF files counter
            'other_files': 0,
            'documents_created': 0,
            'images_extracted': 0,
            'processing_errors': 0,
            'advanced_parsing_used': 0,
            'fallback_used': 0,
            'pdf_processing_used': 0  # NEW: PDF processing counter
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
        Scan directory for files and categorize them (including PDF)
        
        Returns:
            dict: Categorized file lists
        """
        file_categories = {
            'docx_files': [],
            'doc_files': [],
            'pdf_files': [],  # NEW: PDF files category
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
                    elif file_ext == '.pdf':  # NEW: PDF file detection
                        file_categories['pdf_files'].append(str(file_path))
                    else:
                        file_categories['other_files'].append(str(file_path))
                    
                    self.loading_stats['total_files_found'] += 1
            
            self.loading_stats['docx_files'] = len(file_categories['docx_files'])
            self.loading_stats['doc_files'] = len(file_categories['doc_files'])
            self.loading_stats['pdf_files'] = len(file_categories['pdf_files'])  # NEW
            self.loading_stats['other_files'] = len(file_categories['other_files'])
            
        except Exception as e:
            print(f"ERROR: Failed to scan directory {self.input_dir}: {e}")
        
        return file_categories
    
    def load_data(self):
        """
        Load and process all documents using advanced parsing (including PDF)
        
        Returns:
            tuple: (documents, loading_stats, failed_files)
        """
        print("?? Advanced Document Loading with Hybrid Processing + PDF Support")
        
        # Scan files
        file_categories = self.scan_files()
        
        if self.loading_stats['total_files_found'] == 0:
            print("?? WARNING: No files found in directory")
            return [], self.loading_stats, []
        
        print(f"   ?? Found: {self.loading_stats['docx_files']} .docx, "
              f"{self.loading_stats['doc_files']} .doc, "
              f"{self.loading_stats['pdf_files']} .pdf, "  # NEW
              f"{self.loading_stats['other_files']} other files")
        
        all_documents = []
        failed_files = []
        
        # Process DOCX files with advanced parsing
        if file_categories['docx_files']:
            print(f"   ?? Processing {len(file_categories['docx_files'])} DOCX files...")
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
                    print(f"   ? ERROR: Failed to process {docx_file}: {e}")
                    failed_files.append(docx_file)
                    self.loading_stats['processing_errors'] += 1
        
        # Process DOC files with legacy conversion
        if file_categories['doc_files']:
            print(f"   ?? Processing {len(file_categories['doc_files'])} DOC files...")
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
                    print(f"   ? ERROR: Failed to process {doc_file}: {e}")
                    failed_files.append(doc_file)
                    self.loading_stats['processing_errors'] += 1
        
        # NEW: Process PDF files with enhanced PDF processor
        if file_categories['pdf_files']:
            print(f"   ?? Processing {len(file_categories['pdf_files'])} PDF files with enhanced processor...")
            
            # Check if PDF processing is enabled
            pdf_enabled = self.config and self.config.is_feature_enabled('enhanced_pdf_processing')
            
            if not pdf_enabled:
                print(f"   ?? WARNING: Enhanced PDF processing is disabled in configuration")
                print(f"   ?? Enable with: ENABLE_ENHANCED_PDF_PROCESSING=true")
                # Still try to process with fallback
            
            for pdf_file in file_categories['pdf_files']:
                try:
                    documents = self.hybrid_processor.process_pdf_file(pdf_file)
                    if documents:
                        all_documents.extend(documents)
                        self.loading_stats['documents_created'] += len(documents)
                        self.loading_stats['pdf_processing_used'] += 1
                        self.loading_stats['advanced_parsing_used'] += 1
                        
                        # Log PDF processing success
                        total_chars = sum(len(doc.text) for doc in documents)
                        print(f"   ? {os.path.basename(pdf_file)}: {total_chars:,} characters extracted")
                    else:
                        failed_files.append(pdf_file)
                        self.loading_stats['processing_errors'] += 1
                        print(f"   ? {os.path.basename(pdf_file)}: No content extracted")
                        
                except Exception as e:
                    print(f"   ? ERROR: Failed to process PDF {pdf_file}: {e}")
                    failed_files.append(pdf_file)
                    self.loading_stats['processing_errors'] += 1
        
        # Process other files with standard processing
        if file_categories['other_files']:
            print(f"   ?? Processing {len(file_categories['other_files'])} other files...")
            
            for other_file in file_categories['other_files']:
                try:
                    # Use simple file reading for non-Word/PDF documents
                    documents = self.hybrid_processor._simple_file_processing(other_file)
                    if documents:
                        all_documents.extend(documents)
                        self.loading_stats['documents_created'] += len(documents)
                        self.loading_stats['fallback_used'] += 1
                    else:
                        failed_files.append(other_file)
                        self.loading_stats['processing_errors'] += 1
                        
                except Exception as e:
                    print(f"   ? ERROR: Failed to process {other_file}: {e}")
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
        
        print(f"   ? SUCCESSFULLY processed: {total_success} documents")
        print(f"   ? FAILED: {total_failed} files")
        print(f"   ?? Success rate: {success_rate:.1f}%")
        
        # NEW: PDF-specific statistics
        if self.loading_stats['pdf_files'] > 0:
            pdf_success_rate = (self.loading_stats['pdf_processing_used'] / self.loading_stats['pdf_files'] * 100)
            print(f"   ?? PDF processing success rate: {pdf_success_rate:.1f}%")
        
        if self.loading_stats['images_extracted'] > 0:
            print(f"   ??? Images extracted and processed: {self.loading_stats['images_extracted']}")
        
        return all_documents, self.loading_stats, failed_files
    
    def get_processing_summary(self):
        """
        Get comprehensive processing summary (including PDF statistics)
        
        Returns:
            dict: Processing summary with detailed statistics
        """
        return {
            'total_files_found': self.loading_stats['total_files_found'],
            'file_breakdown': {
                'docx_files': self.loading_stats['docx_files'],
                'doc_files': self.loading_stats['doc_files'], 
                'pdf_files': self.loading_stats['pdf_files'],  # NEW
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
                'pdf_processing': self.loading_stats['pdf_processing_used'],  # NEW
                'fallback_processing': self.loading_stats['fallback_used']
            },
            'features_enabled': {
                'advanced_parsing': self.hybrid_processor.advanced_parsing_enabled,
                'image_extraction': self.hybrid_processor.extract_images,
                'structure_preservation': self.hybrid_processor.preserve_structure,
                'table_extraction': self.hybrid_processor.extract_tables,
                'hybrid_processing': self.hybrid_processor.hybrid_processing,
                'pdf_processing': hasattr(self.hybrid_processor, 'pdf_processor') and self.hybrid_processor.pdf_processor is not None  # NEW
            }
        }


def scan_directory_files(directory, recursive=True, include_pdf=True):
    """
    Scan directory and return comprehensive file statistics (including PDF)
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan recursively
        include_pdf: Whether to include PDF files in statistics
    
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
            'pdf_files': 0,  # NEW: PDF files counter
            'image_files': 0,
            'other_files': 0,
            'total_size': 0,
            'large_files': [],
            'problematic_files': [],
            'advanced_parsing_candidates': 0,
            'pdf_processing_candidates': 0,  # NEW: PDF processing candidates
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
                    elif file_ext == '.pdf' and include_pdf:  # NEW: PDF categorization
                        stats['pdf_files'] += 1
                elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                    stats['image_files'] += 1
                else:
                    stats['other_files'] += 1
                
                # Count files that support advanced parsing
                if file_ext in ['.docx', '.doc']:
                    stats['advanced_parsing_candidates'] += 1
                
                # NEW: Count files that support PDF processing
                if file_ext == '.pdf' and include_pdf:
                    stats['pdf_processing_candidates'] += 1
                
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


def analyze_pdf_files_in_directory(directory, recursive=True, max_analyze=10):
    """
    NEW: Analyze PDF files in directory for processing strategy
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan recursively
        max_analyze: Maximum number of files to analyze in detail
    
    Returns:
        dict: PDF analysis results
    """
    try:
        from document_parsers import EnhancedPDFProcessor
        
        # Find all PDF files
        pdf_files = []
        if recursive:
            file_iterator = Path(directory).rglob('*.pdf')
        else:
            file_iterator = Path(directory).glob('*.pdf')
        
        for pdf_file in file_iterator:
            if pdf_file.is_file():
                pdf_files.append(str(pdf_file))
        
        if not pdf_files:
            return {'total_pdfs': 0, 'message': 'No PDF files found'}
        
        # Create PDF processor for analysis
        pdf_processor = EnhancedPDFProcessor()
        
        analysis_results = {
            'total_pdfs': len(pdf_files),
            'analyzed_pdfs': 0,
            'pdf_types': {
                'digital': 0,
                'scanned': 0,
                'structured': 0,
                'mixed': 0,
                'unknown': 0
            },
            'recommended_methods': {
                'pymupdf': 0,
                'pdfplumber': 0,
                'ocr': 0,
                'hybrid': 0
            },
            'size_analysis': {
                'total_pages': 0,
                'avg_pages_per_pdf': 0,
                'large_pdfs': [],  # PDFs with >50 pages
                'small_pdfs': []   # PDFs with <5 pages
            },
            'processing_estimates': {
                'estimated_total_time': 0,
                'fast_processing_files': 0,
                'slow_processing_files': 0
            }
        }
        
        # Analyze sample of PDF files
        sample_size = min(max_analyze, len(pdf_files))
        print(f"?? Analyzing {sample_size} PDF files for processing strategy...")
        
        for i, pdf_file in enumerate(pdf_files[:sample_size]):
            try:
                # Analyze PDF type
                pdf_analysis = pdf_processor.detect_pdf_type(pdf_file)
                
                # Update statistics
                pdf_type = pdf_analysis.get('type', 'unknown')
                analysis_results['pdf_types'][pdf_type] += 1
                
                recommended_method = pdf_analysis.get('recommended_method', 'pymupdf')
                analysis_results['recommended_methods'][recommended_method] += 1
                
                page_count = pdf_analysis.get('page_count', 0)
                analysis_results['size_analysis']['total_pages'] += page_count
                
                # Categorize by size
                if page_count > 50:
                    analysis_results['size_analysis']['large_pdfs'].append({
                        'file': os.path.basename(pdf_file),
                        'pages': page_count,
                        'type': pdf_type
                    })
                elif page_count < 5:
                    analysis_results['size_analysis']['small_pdfs'].append({
                        'file': os.path.basename(pdf_file),
                        'pages': page_count,
                        'type': pdf_type
                    })
                
                # Estimate processing time
                if recommended_method == 'ocr':
                    analysis_results['processing_estimates']['slow_processing_files'] += 1
                else:
                    analysis_results['processing_estimates']['fast_processing_files'] += 1
                
                analysis_results['analyzed_pdfs'] += 1
                
                # Progress indicator
                if (i + 1) % 5 == 0:
                    print(f"   ?? Analyzed {i + 1}/{sample_size} PDFs...")
                
            except Exception as e:
                print(f"   ?? Failed to analyze {os.path.basename(pdf_file)}: {e}")
        
        # Calculate averages and estimates
        if analysis_results['analyzed_pdfs'] > 0:
            analysis_results['size_analysis']['avg_pages_per_pdf'] = (
                analysis_results['size_analysis']['total_pages'] / analysis_results['analyzed_pdfs']
            )
            
            # Estimate total processing time (rough)
            fast_files = analysis_results['processing_estimates']['fast_processing_files']
            slow_files = analysis_results['processing_estimates']['slow_processing_files']
            
            # Extrapolate to all files
            total_fast = int((fast_files / analysis_results['analyzed_pdfs']) * len(pdf_files))
            total_slow = int((slow_files / analysis_results['analyzed_pdfs']) * len(pdf_files))
            
            # Rough time estimates (seconds per file)
            estimated_time = (total_fast * 2) + (total_slow * 10)  # 2s for fast, 10s for slow
            analysis_results['processing_estimates']['estimated_total_time'] = estimated_time
        
        return analysis_results
        
    except Exception as e:
        return {'error': str(e), 'total_pdfs': 0}


def print_pdf_analysis_summary(analysis_results):
    """
    NEW: Print summary of PDF analysis results
    
    Args:
        analysis_results: Results from analyze_pdf_files_in_directory
    """
    if 'error' in analysis_results:
        print(f"? PDF analysis failed: {analysis_results['error']}")
        return
    
    if analysis_results['total_pdfs'] == 0:
        print("?? No PDF files found for analysis")
        return
    
    print(f"\n?? PDF ANALYSIS SUMMARY:")
    print(f"   ?? Total PDF files: {analysis_results['total_pdfs']}")
    print(f"   ?? Analyzed: {analysis_results['analyzed_pdfs']}")
    
    # PDF types distribution
    pdf_types = analysis_results['pdf_types']
    print(f"\n?? PDF Types:")
    for pdf_type, count in pdf_types.items():
        if count > 0:
            percentage = (count / analysis_results['analyzed_pdfs'] * 100)
            print(f"   {pdf_type.title()}: {count} files ({percentage:.1f}%)")
    
    # Recommended methods
    methods = analysis_results['recommended_methods']
    print(f"\n?? Recommended Processing Methods:")
    for method, count in methods.items():
        if count > 0:
            percentage = (count / analysis_results['analyzed_pdfs'] * 100)
            print(f"   {method}: {count} files ({percentage:.1f}%)")
    
    # Size analysis
    size_analysis = analysis_results['size_analysis']
    print(f"\n?? Size Analysis:")
    print(f"   Total pages: {size_analysis['total_pages']}")
    print(f"   Average pages per PDF: {size_analysis['avg_pages_per_pdf']:.1f}")
    
    if size_analysis['large_pdfs']:
        print(f"   Large PDFs (>50 pages): {len(size_analysis['large_pdfs'])}")
        for pdf in size_analysis['large_pdfs'][:3]:  # Show first 3
            print(f"     - {pdf['file']}: {pdf['pages']} pages ({pdf['type']})")
    
    if size_analysis['small_pdfs']:
        print(f"   Small PDFs (<5 pages): {len(size_analysis['small_pdfs'])}")
    
    # Processing estimates
    estimates = analysis_results['processing_estimates']
    print(f"\n?? Processing Estimates:")
    print(f"   Fast processing files: {estimates['fast_processing_files']}")
    print(f"   Slow processing files: {estimates['slow_processing_files']}")
    if estimates['estimated_total_time'] > 0:
        total_time = estimates['estimated_total_time']
        if total_time < 60:
            print(f"   Estimated total time: {total_time:.0f} seconds")
        elif total_time < 3600:
            print(f"   Estimated total time: {total_time/60:.1f} minutes")
        else:
            print(f"   Estimated total time: {total_time/3600:.1f} hours")


def get_enhanced_directory_summary(directory, recursive=True, analyze_pdfs=True):
    """
    NEW: Get enhanced directory summary including PDF analysis
    
    Args:
        directory: Directory to analyze
        recursive: Whether to scan recursively
        analyze_pdfs: Whether to perform detailed PDF analysis
    
    Returns:
        dict: Complete directory analysis
    """
    print(f"?? Enhanced Directory Analysis: {directory}")
    
    # Basic file statistics
    stats = scan_directory_files(directory, recursive, include_pdf=True)
    
    if 'error' in stats:
        return {'error': stats['error']}
    
    # PDF-specific analysis
    pdf_analysis = {}
    if analyze_pdfs and stats.get('pdf_files', 0) > 0:
        print(f"?? Found {stats['pdf_files']} PDF files, performing detailed analysis...")
        pdf_analysis = analyze_pdf_files_in_directory(directory, recursive, max_analyze=10)
    
    # Combine results
    enhanced_summary = {
        'directory': directory,
        'scan_recursive': recursive,
        'basic_stats': stats,
        'pdf_analysis': pdf_analysis,
        'processing_recommendations': _generate_processing_recommendations(stats, pdf_analysis)
    }
    
    return enhanced_summary


def _generate_processing_recommendations(stats, pdf_analysis):
    """
    Generate processing recommendations based on file analysis
    
    Args:
        stats: Basic file statistics
        pdf_analysis: PDF-specific analysis
    
    Returns:
        list: Processing recommendations
    """
    recommendations = []
    
    # Document type recommendations
    if stats.get('docx_files', 0) > 0:
        recommendations.append(f"? {stats['docx_files']} DOCX files - advanced parsing recommended")
    
    if stats.get('doc_files', 0) > 0:
        recommendations.append(f"?? {stats['doc_files']} DOC files - auto-conversion recommended")
    
    if stats.get('pdf_files', 0) > 0:
        recommendations.append(f"?? {stats['pdf_files']} PDF files - enhanced PDF processing recommended")
        
        # PDF-specific recommendations
        if pdf_analysis and 'pdf_types' in pdf_analysis:
            scanned_pdfs = pdf_analysis['pdf_types'].get('scanned', 0)
            if scanned_pdfs > 0:
                recommendations.append(f"?? {scanned_pdfs} scanned PDFs detected - OCR fallback needed")
            
            structured_pdfs = pdf_analysis['pdf_types'].get('structured', 0)
            if structured_pdfs > 0:
                recommendations.append(f"?? {structured_pdfs} structured PDFs - table extraction beneficial")
    
    # Performance recommendations
    total_large_files = len(stats.get('large_files', []))
    if total_large_files > 0:
        recommendations.append(f"? {total_large_files} large files (>10MB) - consider batch processing")
    
    # Feature recommendations
    if stats.get('image_files', 0) > 0:
        recommendations.append(f"??? {stats['image_files']} image files - OCR processing recommended")
    
    return recommendations


def print_enhanced_directory_summary(enhanced_summary):
    """
    Print comprehensive directory summary including PDF analysis
    
    Args:
        enhanced_summary: Results from get_enhanced_directory_summary
    """
    if 'error' in enhanced_summary:
        print(f"? Directory analysis failed: {enhanced_summary['error']}")
        return
    
    stats = enhanced_summary['basic_stats']
    pdf_analysis = enhanced_summary.get('pdf_analysis', {})
    
    print(f"\n?? ENHANCED DIRECTORY SUMMARY")
    print(f"Directory: {enhanced_summary['directory']}")
    print(f"Recursive scan: {'Yes' if enhanced_summary['scan_recursive'] else 'No'}")
    print("=" * 60)
    
    # Basic statistics
    print(f"?? File Statistics:")
    print(f"   Total files: {stats['total_files']:,}")
    print(f"   Total size: {stats['total_size_mb']:.1f} MB")
    print(f"   Document files: {stats['document_files']:,}")
    print(f"     - DOCX: {stats['docx_files']:,}")
    print(f"     - DOC: {stats['doc_files']:,}")
    print(f"     - PDF: {stats['pdf_files']:,}")
    print(f"   Text files: {stats['text_files']:,}")
    print(f"   Image files: {stats['image_files']:,}")
    print(f"   Other files: {stats['other_files']:,}")
    
    # Processing candidates
    print(f"\n?? Processing Candidates:")
    print(f"   Advanced parsing candidates: {stats['advanced_parsing_candidates']:,}")
    print(f"   PDF processing candidates: {stats['pdf_processing_candidates']:,}")
    
    # PDF analysis
    if pdf_analysis and 'total_pdfs' in pdf_analysis and pdf_analysis['total_pdfs'] > 0:
        print_pdf_analysis_summary(pdf_analysis)
    
    # Recommendations
    recommendations = enhanced_summary.get('processing_recommendations', [])
    if recommendations:
        print(f"\n?? Processing Recommendations:")
        for rec in recommendations:
            print(f"   {rec}")
    
    print("=" * 60)