#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced PDF processor module for RAG Document Indexer
High-performance PDF text extraction using best practices from 2024 research
Hybrid approach: PyMuPDF for speed + pdfplumber for complex structures + OCR for scanned content
"""

import os
import io
import time
from pathlib import Path
from datetime import datetime
from llama_index.core import Document

# Core PDF processing imports
try:
    import fitz  # PyMuPDF - fastest PDF processor
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber  # Best for tables and complex structures
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Utility imports
from file_utils_core import (
    clean_content_from_null_bytes, 
    clean_metadata_recursive,
    validate_file_path,
    get_file_info
)


class PDFProcessor:
    """Enhanced PDF processor using hybrid approach for optimal text extraction"""
    
    def __init__(self, config=None):
        """
        Initialize PDF processor with configuration
        
        Args:
            config: Configuration object with PDF settings
        """
        self.config = config
        
        # Load settings from config or use defaults
        if config:
            self.chunk_size = getattr(config, 'PDF_CHUNK_SIZE', 2048)
            self.preserve_structure = getattr(config, 'PDF_PRESERVE_STRUCTURE', True)
            self.min_section_length = getattr(config, 'PDF_MIN_SECTION_LENGTH', 200)
            self.header_detection = getattr(config, 'PDF_HEADER_DETECTION', True)
            self.footer_cleanup = getattr(config, 'PDF_FOOTER_CLEANUP', True)
            self.enable_ocr_fallback = getattr(config, 'ENABLE_OCR', True)
        else:
            # Default settings
            self.chunk_size = 2048
            self.preserve_structure = True
            self.min_section_length = 200
            self.header_detection = True
            self.footer_cleanup = True
            self.enable_ocr_fallback = True
        
        # Check library availability
        self.libraries_available = {
            'pymupdf': PYMUPDF_AVAILABLE,
            'pdfplumber': PDFPLUMBER_AVAILABLE,
            'pdf2image': PDF2IMAGE_AVAILABLE
        }
        
        # Processing statistics
        self.stats = {
            'files_processed': 0,
            'total_pages': 0,
            'text_extracted_chars': 0,
            'ocr_pages': 0,
            'structured_pages': 0,
            'processing_time': 0,
            'method_usage': {
                'pymupdf_primary': 0,
                'pdfplumber_tables': 0,
                'ocr_fallback': 0,
                'failed_extractions': 0
            }
        }
        
        # OCR processor (will be injected if needed)
        self.ocr_processor = None
        
        self._print_availability_status()
    
    def _print_availability_status(self):
        """Print status of available PDF processing libraries"""
        print(f"?? PDF Processor initialized:")
        print(f"   PyMuPDF (speed): {'?' if self.libraries_available['pymupdf'] else '?'}")
        print(f"   pdfplumber (tables): {'?' if self.libraries_available['pdfplumber'] else '?'}")
        print(f"   pdf2image (OCR): {'?' if self.libraries_available['pdf2image'] else '?'}")
        
        if not any(self.libraries_available.values()):
            print("   ?? WARNING: No PDF libraries available!")
            print("   Install with: pip install PyMuPDF pdfplumber pdf2image")
    
    def set_ocr_processor(self, ocr_processor):
        """
        Set OCR processor for fallback processing
        
        Args:
            ocr_processor: OCR processor instance
        """
        self.ocr_processor = ocr_processor
    
    def detect_pdf_type(self, file_path):
        """
        Detect PDF type to choose optimal processing strategy
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            dict: PDF analysis results
        """
        analysis = {
            'type': 'unknown',
            'has_text': False,
            'has_images': False,
            'has_tables': False,
            'is_scanned': False,
            'page_count': 0,
            'text_coverage': 0.0,
            'recommended_method': 'pymupdf'
        }
        
        if not self.libraries_available['pymupdf']:
            analysis['recommended_method'] = 'fallback'
            return analysis
        
        try:
            # Quick analysis with PyMuPDF
            doc = fitz.open(file_path)
            analysis['page_count'] = len(doc)
            
            # Sample first few pages for analysis
            sample_pages = min(3, len(doc))
            total_text_length = 0
            total_char_count = 0
            
            for page_num in range(sample_pages):
                page = doc[page_num]
                
                # Extract text to check coverage
                text = page.get_text()
                total_text_length += len(text.strip())
                
                # Count characters for density calculation
                char_count = len([c for c in text if c.isalnum()])
                total_char_count += char_count
                
                # Check for images
                image_list = page.get_images()
                if image_list:
                    analysis['has_images'] = True
                
                # Basic table detection (look for table-like structures)
                if self._detect_table_patterns(text):
                    analysis['has_tables'] = True
            
            doc.close()
            
            # Determine PDF characteristics
            analysis['has_text'] = total_text_length > 50
            analysis['text_coverage'] = total_char_count / (sample_pages * 1000) if sample_pages > 0 else 0
            
            # Classify PDF type
            if analysis['text_coverage'] < 0.1:
                analysis['type'] = 'scanned'
                analysis['is_scanned'] = True
                analysis['recommended_method'] = 'ocr'
            elif analysis['has_tables'] and self.libraries_available['pdfplumber']:
                analysis['type'] = 'structured'
                analysis['recommended_method'] = 'pdfplumber'
            elif analysis['has_text']:
                analysis['type'] = 'digital'
                analysis['recommended_method'] = 'pymupdf'
            else:
                analysis['type'] = 'mixed'
                analysis['recommended_method'] = 'hybrid'
            
        except Exception as e:
            print(f"   WARNING: PDF analysis failed: {e}")
            analysis['recommended_method'] = 'fallback'
        
        return analysis
    
    def _detect_table_patterns(self, text):
        """
        Simple heuristic to detect table-like content
        
        Args:
            text: Text content to analyze
        
        Returns:
            bool: True if table patterns detected
        """
        if not text:
            return False
        
        lines = text.split('\n')
        if len(lines) < 3:
            return False
        
        # Look for patterns indicating tables
        tab_separated_lines = sum(1 for line in lines if '\t' in line)
        space_separated_lines = sum(1 for line in lines if len(line.split()) > 4)
        
        # If more than 30% of lines look table-like
        table_ratio = (tab_separated_lines + space_separated_lines) / len(lines)
        return table_ratio > 0.3
    
    def extract_text_pymupdf(self, file_path):
        """
        Extract text using enhanced PyMuPDF extraction methods
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            tuple: (text_content, extraction_info)
        """
        if not self.libraries_available['pymupdf']:
            return "", {'error': 'PyMuPDF not available'}
        
        try:
            doc = fitz.open(file_path)
            text_parts = []
            extraction_info = {
                'method': 'enhanced_pymupdf',
                'pages_processed': 0,
                'total_chars': 0,
                'processing_time': 0,
                'extraction_modes_used': []
            }
            
            start_time = time.time()
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Enhanced extraction: try multiple methods and use the best
                extraction_methods = []
                
                # Method 1: Standard text extraction
                text1 = page.get_text("text") if self.preserve_structure else page.get_text()
                extraction_methods.append(('standard', text1))
                
                # Method 2: Blocks extraction
                blocks = page.get_text("blocks")
                text2 = ""
                if isinstance(blocks, list):
                    for block in blocks:
                        if isinstance(block, tuple) and len(block) > 4:
                            text2 += str(block[4]) + " "
                        elif isinstance(block, dict) and 'text' in block:
                            text2 += block['text'] + " "
                extraction_methods.append(('blocks', text2))
                
                # Method 3: Words extraction
                words = page.get_text("words")
                text3 = ""
                if isinstance(words, list):
                    text3 = " ".join([str(word[4]) for word in words 
                                    if isinstance(word, tuple) and len(word) > 4])
                extraction_methods.append(('words', text3))
                
                # Method 4: Dictionary extraction
                text_dict = page.get_text("dict")
                text4 = self._extract_from_dict(text_dict)
                extraction_methods.append(('dict', text4))
                
                # Choose the best extraction method for this page
                best_text = ""
                best_method = "standard"
                max_chars = 0
                
                for method_name, text in extraction_methods:
                    if text and len(text.strip()) > max_chars:
                        best_text = text.strip()
                        best_method = method_name
                        max_chars = len(best_text)
                
                if best_text:
                    # Clean and process text
                    cleaned_text = clean_content_from_null_bytes(best_text)
                    
                    # Optional header/footer cleanup
                    if self.footer_cleanup:
                        cleaned_text = self._clean_headers_footers(cleaned_text, page_num)
                    
                    text_parts.append(cleaned_text)
                    extraction_info['total_chars'] += len(cleaned_text)
                    
                    # Track which method worked best
                    if best_method not in extraction_info['extraction_modes_used']:
                        extraction_info['extraction_modes_used'].append(best_method)
                
                extraction_info['pages_processed'] += 1
            
            doc.close()
            
            extraction_info['processing_time'] = time.time() - start_time
            
            # Combine text with proper spacing
            full_text = '\n\n'.join(text_parts)
            
            # Update statistics
            self.stats['method_usage']['pymupdf_primary'] += 1
            
            # Enhanced logging
            if extraction_info['extraction_modes_used']:
                print(f"   ?? Used extraction modes: {', '.join(extraction_info['extraction_modes_used'])}")
            
            return full_text, extraction_info
            
        except Exception as e:
            return "", {'error': str(e), 'method': 'enhanced_pymupdf'}
    
    def _extract_from_dict(self, text_dict):
        """
        Extract text from PyMuPDF dictionary format (enhanced method)
        
        Args:
            text_dict: Dictionary from get_text("dict")
        
        Returns:
            str: Extracted text
        """
        text_parts = []
        
        try:
            if isinstance(text_dict, dict) and 'blocks' in text_dict:
                for block in text_dict['blocks']:
                    if isinstance(block, dict) and 'lines' in block:
                        for line in block['lines']:
                            if isinstance(line, dict) and 'spans' in line:
                                line_text = ""
                                for span in line['spans']:
                                    if isinstance(span, dict) and 'text' in span:
                                        line_text += span['text']
                                if line_text.strip():
                                    text_parts.append(line_text.strip())
        except Exception:
            pass
        
        return ' '.join(text_parts)
    
    def extract_text_pdfplumber(self, file_path):
        """
        Extract text using pdfplumber (best for tables and complex structures)
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            tuple: (text_content, extraction_info)
        """
        if not self.libraries_available['pdfplumber']:
            return "", {'error': 'pdfplumber not available'}
        
        try:
            text_parts = []
            extraction_info = {
                'method': 'pdfplumber',
                'pages_processed': 0,
                'tables_found': 0,
                'total_chars': 0,
                'processing_time': 0
            }
            
            start_time = time.time()
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = ""
                    
                    # Extract regular text
                    if self.preserve_structure:
                        text = page.extract_text(layout=True)
                    else:
                        text = page.extract_text()
                    
                    if text:
                        page_text += text
                    
                    # Extract tables separately for better formatting
                    try:
                        tables = page.extract_tables()
                        if tables:
                            extraction_info['tables_found'] += len(tables)
                            for table in tables:
                                table_text = self._format_table_text(table)
                                if table_text:
                                    page_text += f"\n\n[TABLE]\n{table_text}\n[/TABLE]\n"
                    except Exception as e:
                        print(f"   WARNING: Table extraction failed on page {page_num + 1}: {e}")
                    
                    if page_text.strip():
                        cleaned_text = clean_content_from_null_bytes(page_text)
                        text_parts.append(cleaned_text)
                        extraction_info['total_chars'] += len(cleaned_text)
                    
                    extraction_info['pages_processed'] += 1
            
            extraction_info['processing_time'] = time.time() - start_time
            
            # Combine text
            full_text = '\n\n'.join(text_parts)
            
            # Update statistics
            self.stats['method_usage']['pdfplumber_tables'] += 1
            
            return full_text, extraction_info
            
        except Exception as e:
            return "", {'error': str(e), 'method': 'pdfplumber'}
    
    def extract_text_ocr_fallback(self, file_path):
        """
        Extract text using OCR fallback (for scanned PDFs)
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            tuple: (text_content, extraction_info)
        """
        if not (self.libraries_available['pdf2image'] and self.ocr_processor):
            return "", {'error': 'OCR fallback not available'}
        
        try:
            extraction_info = {
                'method': 'ocr_fallback',
                'pages_processed': 0,
                'total_chars': 0,
                'processing_time': 0
            }
            
            start_time = time.time()
            
            # Convert PDF pages to images
            images = convert_from_path(file_path, dpi=300, fmt='jpeg')
            text_parts = []
            
            for page_num, image in enumerate(images):
                try:
                    # Save image temporarily for OCR processing
                    temp_image_path = f"/tmp/pdf_page_{page_num}.jpg"
                    image.save(temp_image_path, 'JPEG')
                    
                    # Process with OCR
                    text = self.ocr_processor.extract_text_from_image(temp_image_path)
                    
                    if text and len(text.strip()) > 20:
                        cleaned_text = clean_content_from_null_bytes(text)
                        text_parts.append(cleaned_text)
                        extraction_info['total_chars'] += len(cleaned_text)
                    
                    # Clean up temporary file
                    if os.path.exists(temp_image_path):
                        os.unlink(temp_image_path)
                    
                    extraction_info['pages_processed'] += 1
                    
                except Exception as e:
                    print(f"   WARNING: OCR failed for page {page_num + 1}: {e}")
            
            extraction_info['processing_time'] = time.time() - start_time
            
            # Combine text
            full_text = '\n\n'.join(text_parts)
            
            # Update statistics
            self.stats['method_usage']['ocr_fallback'] += 1
            
            return full_text, extraction_info
            
        except Exception as e:
            return "", {'error': str(e), 'method': 'ocr_fallback'}
    
    def _format_table_text(self, table):
        """
        Format extracted table data into readable text
        
        Args:
            table: Table data from pdfplumber
        
        Returns:
            str: Formatted table text
        """
        if not table:
            return ""
        
        try:
            formatted_rows = []
            for row in table:
                if row:
                    # Clean and join cells
                    cells = [str(cell).strip() if cell else "" for cell in row]
                    row_text = " | ".join(cells)
                    if row_text.strip():
                        formatted_rows.append(row_text)
            
            return '\n'.join(formatted_rows)
        except Exception as e:
            print(f"   WARNING: Table formatting failed: {e}")
            return ""
    
    def _clean_headers_footers(self, text, page_num):
        """
        Basic header/footer cleanup
        
        Args:
            text: Text content
            page_num: Page number
        
        Returns:
            str: Cleaned text
        """
        if not text:
            return text
        
        lines = text.split('\n')
        if len(lines) < 5:
            return text
        
        # Remove common header/footer patterns
        cleaned_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip very short lines at top/bottom
            if (i < 2 or i >= len(lines) - 2) and len(line) < 50:
                # Check if it's just page numbers or headers
                if line.isdigit() or 'page' in line.lower():
                    continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def process_pdf_file(self, file_path):
        """
        Process a single PDF file using optimal strategy
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            list: List of Document objects
        """
        print(f"?? Processing PDF: {os.path.basename(file_path)}")
        
        # Validate file
        is_valid, error_msg = validate_file_path(file_path)
        if not is_valid:
            print(f"   ? Invalid file: {error_msg}")
            return []
        
        # Get file info
        file_info = get_file_info(file_path)
        if 'error' in file_info:
            print(f"   ? File info error: {file_info['error']}")
            return []
        
        # Analyze PDF to choose strategy
        pdf_analysis = self.detect_pdf_type(file_path)
        print(f"   ?? PDF type: {pdf_analysis['type']} ({pdf_analysis['page_count']} pages)")
        print(f"   ?? Strategy: {pdf_analysis['recommended_method']}")
        
        # Extract text using optimal method
        extraction_method = pdf_analysis['recommended_method']
        text_content = ""
        extraction_info = {}
        
        if extraction_method == 'pymupdf' and self.libraries_available['pymupdf']:
            text_content, extraction_info = self.extract_text_pymupdf(file_path)
        elif extraction_method == 'pdfplumber' and self.libraries_available['pdfplumber']:
            text_content, extraction_info = self.extract_text_pdfplumber(file_path)
        elif extraction_method == 'ocr' and self.ocr_processor:
            text_content, extraction_info = self.extract_text_ocr_fallback(file_path)
        else:
            # Fallback chain
            if self.libraries_available['pymupdf']:
                text_content, extraction_info = self.extract_text_pymupdf(file_path)
            elif self.libraries_available['pdfplumber']:
                text_content, extraction_info = self.extract_text_pdfplumber(file_path)
            else:
                print(f"   ? No PDF processing libraries available")
                self.stats['method_usage']['failed_extractions'] += 1
                return []
        
        # Check extraction result
        if 'error' in extraction_info:
            print(f"   ? Extraction failed: {extraction_info['error']}")
            self.stats['method_usage']['failed_extractions'] += 1
            return []
        
        if not text_content or len(text_content.strip()) < 50:  # Reduced threshold for certificates
            print(f"   ?? Insufficient text extracted ({len(text_content)} chars)")
            if self.enable_ocr_fallback and self.ocr_processor and extraction_method != 'ocr':
                print(f"   ?? Trying OCR fallback...")
                text_content, ocr_info = self.extract_text_ocr_fallback(file_path)
                if text_content and len(text_content.strip()) >= 50:  # Lower threshold for OCR too
                    extraction_info = ocr_info
                else:
                    # For certificates and short documents, accept even shorter content
                    if len(text_content.strip()) >= 20:
                        print(f"   ?? Accepting short content (likely certificate): {len(text_content)} chars")
                    else:
                        self.stats['method_usage']['failed_extractions'] += 1
                        return []
            else:
                # Accept short content for certificates
                if len(text_content.strip()) >= 20:
                    print(f"   ?? Accepting short content (likely certificate): {len(text_content)} chars")
                else:
                    self.stats['method_usage']['failed_extractions'] += 1
                    return []
        
        # Create document with enhanced metadata
        clean_file_path = clean_content_from_null_bytes(str(file_path))
        clean_file_name = clean_content_from_null_bytes(os.path.basename(file_path))
        
        metadata = {
            'file_path': clean_file_path,
            'file_name': clean_file_name,
            'file_type': 'pdf',
            'file_size': file_info['size'],
            'pdf_analysis': pdf_analysis,
            'extraction_info': extraction_info,
            'content_length': len(text_content),
            'processing_timestamp': datetime.now().isoformat(),
            'processor_version': 'enhanced_pdf_processor_v1.0'
        }
        
        # Clean metadata
        clean_metadata = clean_metadata_recursive(metadata)
        
        document = Document(
            text=text_content,
            metadata=clean_metadata
        )
        
        # Update statistics
        self.stats['files_processed'] += 1
        self.stats['total_pages'] += pdf_analysis['page_count']
        self.stats['text_extracted_chars'] += len(text_content)
        self.stats['processing_time'] += extraction_info.get('processing_time', 0)
        
        if extraction_info['method'] == 'ocr_fallback':
            self.stats['ocr_pages'] += extraction_info.get('pages_processed', 0)
        else:
            self.stats['structured_pages'] += extraction_info.get('pages_processed', 0)
        
        print(f"   ? SUCCESS: {len(text_content)} characters extracted in {extraction_info.get('processing_time', 0):.2f}s")
        
        return [document]
    
    def get_processing_stats(self):
        """Get comprehensive processing statistics"""
        return {
            'files_processed': self.stats['files_processed'],
            'total_pages': self.stats['total_pages'],
            'text_extracted_chars': self.stats['text_extracted_chars'],
            'ocr_pages': self.stats['ocr_pages'],
            'structured_pages': self.stats['structured_pages'],
            'processing_time': self.stats['processing_time'],
            'method_usage': self.stats['method_usage'].copy(),
            'average_chars_per_page': self.stats['text_extracted_chars'] / max(self.stats['total_pages'], 1),
            'average_processing_speed': self.stats['total_pages'] / max(self.stats['processing_time'], 0.001),
            'libraries_available': self.libraries_available.copy()
        }
    
    def print_processing_summary(self):
        """Print processing summary"""
        stats = self.get_processing_stats()
        
        print(f"\n?? PDF Processing Summary:")
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Total pages: {stats['total_pages']}")
        print(f"   Text extracted: {stats['text_extracted_chars']:,} characters")
        print(f"   Processing time: {stats['processing_time']:.2f}s")
        print(f"   Average speed: {stats['average_processing_speed']:.1f} pages/sec")
        
        print(f"\n?? Method Usage:")
        for method, count in stats['method_usage'].items():
            print(f"   {method.replace('_', ' ').title()}: {count}")
        
        if stats['ocr_pages'] > 0:
            print(f"\n?? OCR Statistics:")
            print(f"   Pages processed with OCR: {stats['ocr_pages']}")
            print(f"   Structured pages: {stats['structured_pages']}")


def create_pdf_processor(config=None):
    """
    Create PDF processor instance
    
    Args:
        config: Configuration object
    
    Returns:
        PDFProcessor: Configured PDF processor
    """
    return PDFProcessor(config)


def check_pdf_processing_capabilities():
    """
    Check available PDF processing capabilities
    
    Returns:
        dict: Capability information
    """
    capabilities = {
        'pymupdf': PYMUPDF_AVAILABLE,
        'pdfplumber': PDFPLUMBER_AVAILABLE,
        'pdf2image': PDF2IMAGE_AVAILABLE,
        'overall_status': 'ready' if PYMUPDF_AVAILABLE else 'limited'
    }
    
    print("?? PDF Processing Capabilities:")
    print(f"   PyMuPDF (fast): {'?' if capabilities['pymupdf'] else '?'}")
    print(f"   pdfplumber (tables): {'?' if capabilities['pdfplumber'] else '?'}")
    print(f"   pdf2image (OCR): {'?' if capabilities['pdf2image'] else '?'}")
    
    if not any([capabilities['pymupdf'], capabilities['pdfplumber']]):
        print("   ?? Install with: pip install PyMuPDF pdfplumber pdf2image")
    
    return capabilities


if __name__ == "__main__":
    # Test PDF processor capabilities
    check_pdf_processing_capabilities()