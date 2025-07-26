#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced PDF processor module for RAG Document Indexer
Comprehensive PDF text extraction using hybrid approach with advanced features:
- Auto-rotation detection for scanned PDFs
- Multiple extraction methods (PyMuPDF, pdfplumber, OCR fallback)
- Intelligent method selection based on PDF type
- Text quality analysis and validation
- Table extraction and structure preservation
"""

import os
import io
import time
import tempfile
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


class EnhancedPDFProcessor:
    """
    Enhanced PDF processor using hybrid approach for optimal text extraction
    Features:
    - Auto-rotation detection for scanned documents
    - Multiple extraction methods with intelligent selection
    - OCR fallback for image-based PDFs
    - Text quality analysis and validation
    - Table extraction and structure preservation
    """
    
    def __init__(self, config=None):
        """
        Initialize Enhanced PDF processor
        
        Args:
            config: Configuration object with PDF settings
        """
        self.config = config
        
        # Load settings from config or use defaults
        if config:
            pdf_settings = config.get_pdf_processing_settings()
            self.chunk_size = pdf_settings.get('chunk_size', 2048)
            self.preserve_structure = pdf_settings.get('preserve_structure', True)
            self.min_section_length = pdf_settings.get('min_section_length', 200)
            self.header_detection = pdf_settings.get('header_detection', True)
            self.footer_cleanup = pdf_settings.get('footer_cleanup', True)
            self.enable_ocr_fallback = pdf_settings.get('enable_ocr_fallback', True)
            self.auto_method_selection = pdf_settings.get('auto_method_selection', True)
            self.prefer_pymupdf = pdf_settings.get('prefer_pymupdf', True)
            self.enable_table_extraction = pdf_settings.get('enable_table_extraction', True)
            self.scanned_threshold = pdf_settings.get('scanned_threshold', 0.1)
            self.table_detection_threshold = pdf_settings.get('table_detection_threshold', 0.3)
            self.min_content_length = pdf_settings.get('min_content_length', 20)
            self.max_pages_for_analysis = pdf_settings.get('max_pages_for_analysis', 3)
            self.enable_content_validation = pdf_settings.get('enable_content_validation', True)
            
            # OCR settings
            self.ocr_dpi = pdf_settings.get('ocr_dpi', 300)
            self.ocr_image_format = pdf_settings.get('ocr_image_format', 'jpeg')
            self.ocr_min_text_length = pdf_settings.get('ocr_min_text_length', 20)
            self.ocr_timeout_per_page = pdf_settings.get('ocr_timeout_per_page', 30)
        else:
            # Default settings
            self.chunk_size = 2048
            self.preserve_structure = True
            self.min_section_length = 200
            self.header_detection = True
            self.footer_cleanup = True
            self.enable_ocr_fallback = True
            self.auto_method_selection = True
            self.prefer_pymupdf = True
            self.enable_table_extraction = True
            self.scanned_threshold = 0.1
            self.table_detection_threshold = 0.3
            self.min_content_length = 20
            self.max_pages_for_analysis = 3
            self.enable_content_validation = True
            self.ocr_dpi = 300
            self.ocr_image_format = 'jpeg'
            self.ocr_min_text_length = 20
            self.ocr_timeout_per_page = 30
        
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
            },
            'rotation_stats': {
                'images_tested': 0,
                'rotations_applied': 0,
                'improvements_found': 0,
                'timeouts': 0
            },
            'quality_analysis': {
                'high_quality_extractions': 0,
                'low_quality_extractions': 0,
                'ocr_improvements': 0
            }
        }
        
        # OCR processor (will be injected if needed)
        self.ocr_processor = None
        
        self._print_initialization_status()
    
    def _print_initialization_status(self):
        """Print initialization status and available features"""
        print(f"📄 Enhanced PDF Processor initialized:")
        print(f"   PyMuPDF (speed): {'✅' if self.libraries_available['pymupdf'] else '❌'}")
        print(f"   pdfplumber (tables): {'✅' if self.libraries_available['pdfplumber'] else '❌'}")
        print(f"   pdf2image (OCR): {'✅' if self.libraries_available['pdf2image'] else '❌'}")
        print(f"   Auto method selection: {'✅' if self.auto_method_selection else '❌'}")
        print(f"   OCR fallback: {'✅' if self.enable_ocr_fallback else '❌'}")
        print(f"   Table extraction: {'✅' if self.enable_table_extraction else '❌'}")
        
        if not any(self.libraries_available.values()):
            print("   ⚠️ WARNING: No PDF libraries available!")
            print("   Install with: pip install PyMuPDF pdfplumber pdf2image")
    
    def set_ocr_processor(self, ocr_processor):
        """
        Set OCR processor for fallback processing
        
        Args:
            ocr_processor: OCR processor instance
        """
        self.ocr_processor = ocr_processor
        print(f"   🤖 OCR processor integrated for enhanced PDF processing")
    
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
            'recommended_method': 'pymupdf',
            'confidence': 0.0
        }
        
        if not self.libraries_available['pymupdf']:
            analysis['recommended_method'] = 'fallback'
            return analysis
        
        try:
            import fitz
            # Quick analysis with PyMuPDF
            doc = fitz.open(file_path)
            analysis['page_count'] = len(doc)
            
            # Sample first few pages for analysis
            sample_pages = min(self.max_pages_for_analysis, len(doc))
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
            
            # Classify PDF type with confidence scoring
            if analysis['text_coverage'] < self.scanned_threshold:
                analysis['type'] = 'scanned'
                analysis['is_scanned'] = True
                analysis['recommended_method'] = 'ocr'
                analysis['confidence'] = 0.9 if analysis['text_coverage'] < 0.05 else 0.7
            elif analysis['has_tables'] and self.libraries_available['pdfplumber'] and self.enable_table_extraction:
                analysis['type'] = 'structured'
                analysis['recommended_method'] = 'pdfplumber'
                analysis['confidence'] = 0.8
            elif analysis['has_text']:
                analysis['type'] = 'digital'
                analysis['recommended_method'] = 'pymupdf' if self.prefer_pymupdf else 'pdfplumber'
                analysis['confidence'] = 0.9
            else:
                analysis['type'] = 'mixed'
                analysis['recommended_method'] = 'hybrid'
                analysis['confidence'] = 0.6
            
        except Exception as e:
            print(f"   WARNING: PDF analysis failed: {e}")
            analysis['recommended_method'] = 'fallback'
            analysis['confidence'] = 0.3
        
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
        
        # If more than threshold of lines look table-like
        table_ratio = (tab_separated_lines + space_separated_lines) / len(lines)
        return table_ratio > self.table_detection_threshold
    
    def extract_text_pymupdf(self, file_path):
        """
        Extract text using enhanced PyMuPDF extraction methods with multiple strategies
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            tuple: (text_content, extraction_info)
        """
        if not self.libraries_available['pymupdf']:
            return "", {'error': 'PyMuPDF not available'}
        
        try:
            import fitz
            doc = fitz.open(file_path)
            text_parts = []
            extraction_info = {
                'method': 'enhanced_pymupdf',
                'pages_processed': 0,
                'total_chars': 0,
                'processing_time': 0,
                'extraction_modes_used': [],
                'quality_score': 0.0
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
                try:
                    blocks = page.get_text("blocks")
                    text2 = ""
                    if isinstance(blocks, list):
                        for block in blocks:
                            if isinstance(block, tuple) and len(block) > 4:
                                text2 += str(block[4]) + " "
                            elif isinstance(block, dict) and 'text' in block:
                                text2 += block['text'] + " "
                    extraction_methods.append(('blocks', text2))
                except:
                    pass
                
                # Method 3: Words extraction
                try:
                    words = page.get_text("words")
                    text3 = ""
                    if isinstance(words, list):
                        text3 = " ".join([str(word[4]) for word in words 
                                        if isinstance(word, tuple) and len(word) > 4])
                    extraction_methods.append(('words', text3))
                except:
                    pass
                
                # Method 4: Dictionary extraction
                try:
                    text_dict = page.get_text("dict")
                    text4 = self._extract_from_dict(text_dict)
                    extraction_methods.append(('dict', text4))
                except:
                    pass
                
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
            
            # Calculate quality score
            extraction_info['quality_score'] = self._calculate_extraction_quality(full_text)
            
            # Update statistics
            self.stats['method_usage']['pymupdf_primary'] += 1
            self.stats['structured_pages'] += extraction_info['pages_processed']
            
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
            import pdfplumber
            text_parts = []
            extraction_info = {
                'method': 'pdfplumber',
                'pages_processed': 0,
                'tables_found': 0,
                'total_chars': 0,
                'processing_time': 0,
                'quality_score': 0.0
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
                    if self.enable_table_extraction:
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
            
            # Calculate quality score
            extraction_info['quality_score'] = self._calculate_extraction_quality(full_text)
            
            # Update statistics
            self.stats['method_usage']['pdfplumber_tables'] += 1
            self.stats['structured_pages'] += extraction_info['pages_processed']
            
            return full_text, extraction_info
            
        except Exception as e:
            return "", {'error': str(e), 'method': 'pdfplumber'}
    
    def extract_text_ocr_fallback(self, file_path):
        """
        Extract text using OCR fallback (for scanned PDFs) with rotation detection
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            tuple: (text_content, extraction_info)
        """
        if not (self.libraries_available['pdf2image'] and self.ocr_processor):
            return "", {'error': 'OCR fallback not available'}
        
        try:
            from pdf2image import convert_from_path
            extraction_info = {
                'method': 'ocr_fallback',
                'pages_processed': 0,
                'total_chars': 0,
                'processing_time': 0,
                'rotations_applied': 0,
                'quality_improvements': 0,
                'quality_score': 0.0
            }
            
            start_time = time.time()
            
            # Convert PDF pages to images
            images = convert_from_path(
                file_path, 
                dpi=self.ocr_dpi, 
                fmt=self.ocr_image_format
            )
            text_parts = []
            
            for page_num, image in enumerate(images):
                try:
                    # Check timeout
                    if time.time() - start_time > self.ocr_timeout_per_page * len(images):
                        print(f"   WARNING: OCR timeout reached, processing remaining pages with basic extraction")
                        break
                    
                    # Save image temporarily for OCR processing
                    with tempfile.NamedTemporaryFile(suffix=f'.{self.ocr_image_format}', delete=False) as temp_file:
                        temp_image_path = temp_file.name
                        image.save(temp_image_path, self.ocr_image_format.upper())
                    
                    # Process with OCR (includes auto-rotation detection)
                    text = self.ocr_processor.extract_text_from_image(temp_image_path)
                    
                    # Clean up temporary file
                    if os.path.exists(temp_image_path):
                        os.unlink(temp_image_path)
                    
                    if text and len(text.strip()) >= self.ocr_min_text_length:
                        cleaned_text = clean_content_from_null_bytes(text)
                        text_parts.append(cleaned_text)
                        extraction_info['total_chars'] += len(cleaned_text)
                        
                        # Track OCR quality improvements (if OCR processor provides this info)
                        if hasattr(self.ocr_processor, 'rotation_stats'):
                            rotation_stats = self.ocr_processor.rotation_stats
                            if rotation_stats.get('rotations_applied', 0) > extraction_info['rotations_applied']:
                                extraction_info['rotations_applied'] = rotation_stats['rotations_applied']
                                extraction_info['quality_improvements'] = rotation_stats.get('improvements_found', 0)
                    
                    extraction_info['pages_processed'] += 1
                    
                except Exception as e:
                    print(f"   WARNING: OCR failed for page {page_num + 1}: {e}")
                    continue
            
            extraction_info['processing_time'] = time.time() - start_time
            
            # Combine text
            full_text = '\n\n'.join(text_parts)
            
            # Calculate quality score
            extraction_info['quality_score'] = self._calculate_extraction_quality(full_text)
            
            # Update statistics
            self.stats['method_usage']['ocr_fallback'] += 1
            self.stats['ocr_pages'] += extraction_info['pages_processed']
            self.stats['rotation_stats']['rotations_applied'] += extraction_info.get('rotations_applied', 0)
            self.stats['rotation_stats']['improvements_found'] += extraction_info.get('quality_improvements', 0)
            
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
        if not text or not self.footer_cleanup:
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
                if line.isdigit() or 'page' in line.lower() or len(line) < 10:
                    continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _calculate_extraction_quality(self, text):
        """
        Calculate quality score for extracted text
        
        Args:
            text: Extracted text
        
        Returns:
            float: Quality score (0.0 to 1.0)
        """
        if not text:
            return 0.0
        
        # Basic quality metrics
        total_chars = len(text)
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        spaces = sum(c.isspace() for c in text)
        
        if total_chars == 0:
            return 0.0
        
        # Calculate ratios
        letter_ratio = letters / total_chars
        readable_ratio = (letters + digits + spaces) / total_chars
        
        # Word-based metrics
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Combine metrics into quality score
        quality_score = (
            letter_ratio * 0.4 +  # Letters are important
            readable_ratio * 0.3 +  # Overall readability
            min(avg_word_length / 6, 1.0) * 0.2 +  # Reasonable word length
            min(len(words) / 100, 1.0) * 0.1  # Sufficient content
        )
        
        return min(quality_score, 1.0)
    
    def _validate_extracted_content(self, text, extraction_info):
        """
        Validate extracted content quality
        
        Args:
            text: Extracted text
            extraction_info: Extraction information
        
        Returns:
            tuple: (is_valid, validation_details)
        """
        if not self.enable_content_validation:
            return True, {'validation': 'disabled'}
        
        validation_details = {
            'length_check': len(text) >= self.min_content_length,
            'quality_score': extraction_info.get('quality_score', 0.0),
            'content_length': len(text),
            'method_used': extraction_info.get('method', 'unknown')
        }
        
        # Basic length check
        if len(text) < self.min_content_length:
            validation_details['failure_reason'] = 'insufficient_content'
            return False, validation_details
        
        # Quality score check
        if extraction_info.get('quality_score', 0.0) < 0.3:
            validation_details['failure_reason'] = 'low_quality'
            return False, validation_details
        
        validation_details['validation_passed'] = True
        return True, validation_details
    
    def process_pdf_file(self, file_path):
        """
        Process a single PDF file using optimal strategy with enhanced features
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            list: List of Document objects
        """
        print(f"   📄 Processing PDF: {os.path.basename(file_path)}")
        
        # Validate file
        is_valid, error_msg = validate_file_path(file_path)
        if not is_valid:
            print(f"   ❌ Invalid file: {error_msg}")
            return []
        
        # Get file info
        file_info = get_file_info(file_path)
        if 'error' in file_info:
            print(f"   ❌ File info error: {file_info['error']}")
            return []
        
        start_processing_time = time.time()
        
        # Analyze PDF to choose strategy
        pdf_analysis = self.detect_pdf_type(file_path)
        print(f"   🔍 PDF type: {pdf_analysis['type']} ({pdf_analysis['page_count']} pages)")
        print(f"   🎯 Strategy: {pdf_analysis['recommended_method']} (confidence: {pdf_analysis['confidence']:.1f})")
        
        # Extract text using optimal method
        extraction_method = pdf_analysis['recommended_method']
        text_content = ""
        extraction_info = {}
        
        # Try primary method
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
                print(f"   ❌ No PDF processing libraries available")
                self.stats['method_usage']['failed_extractions'] += 1
                return []
        
        # Check extraction result
        if 'error' in extraction_info:
            print(f"   ❌ Extraction failed: {extraction_info['error']}")
            self.stats['method_usage']['failed_extractions'] += 1
            return []
        
        # Validate content
        is_valid, validation_details = self._validate_extracted_content(text_content, extraction_info)
        
        if not is_valid:
            print(f"   ⚠️ Primary extraction insufficient: {validation_details.get('failure_reason', 'unknown')}")
            
            # Try OCR fallback if available and not already used
            if self.enable_ocr_fallback and self.ocr_processor and extraction_method != 'ocr':
                print(f"   🔄 Trying OCR fallback...")
                fallback_text, ocr_info = self.extract_text_ocr_fallback(file_path)
                
                # Validate OCR result
                ocr_valid, ocr_validation = self._validate_extracted_content(fallback_text, ocr_info)
                
                if ocr_valid and len(fallback_text.strip()) > len(text_content.strip()):
                    print(f"   ✅ OCR fallback successful: {len(fallback_text)} chars")
                    text_content = fallback_text
                    extraction_info = ocr_info
                    extraction_info['fallback_used'] = True
                    self.stats['quality_analysis']['ocr_improvements'] += 1
                elif len(text_content.strip()) >= self.min_content_length:
                    # Accept original if it meets minimum requirements
                    print(f"   ⚠️ Using original extraction despite low quality: {len(text_content)} chars")
                else:
                    print(f"   ❌ All extraction methods failed")
                    self.stats['method_usage']['failed_extractions'] += 1
                    return []
            elif len(text_content.strip()) >= self.min_content_length:
                # Accept content if it meets minimum requirements
                print(f"   ⚠️ Using low-quality extraction: {len(text_content)} chars")
            else:
                print(f"   ❌ Extraction failed: insufficient content")
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
            'validation_details': validation_details,
            'content_length': len(text_content),
            'processing_timestamp': datetime.now().isoformat(),
            'processor_version': 'enhanced_pdf_processor_v2.0',
            'processing_time': time.time() - start_processing_time
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
        
        # Quality tracking
        quality_score = extraction_info.get('quality_score', 0.0)
        if quality_score >= 0.7:
            self.stats['quality_analysis']['high_quality_extractions'] += 1
        else:
            self.stats['quality_analysis']['low_quality_extractions'] += 1
        
        print(f"   ✅ SUCCESS: {len(text_content)} characters extracted in {extraction_info.get('processing_time', 0):.2f}s")
        print(f"   📊 Quality score: {quality_score:.2f}, Method: {extraction_info.get('method', 'unknown')}")
        
        return [document]
    
    def get_processing_stats(self):
        """
        Get comprehensive processing statistics
        
        Returns:
            dict: Detailed processing statistics
        """
        total_pages = max(self.stats['total_pages'], 1)
        total_time = max(self.stats['processing_time'], 0.001)
        
        stats = {
            'files_processed': self.stats['files_processed'],
            'total_pages': self.stats['total_pages'],
            'text_extracted_chars': self.stats['text_extracted_chars'],
            'ocr_pages': self.stats['ocr_pages'],
            'structured_pages': self.stats['structured_pages'],
            'processing_time': self.stats['processing_time'],
            'method_usage': self.stats['method_usage'].copy(),
            'rotation_stats': self.stats['rotation_stats'].copy(),
            'quality_analysis': self.stats['quality_analysis'].copy(),
            
            # Calculated metrics
            'average_chars_per_page': self.stats['text_extracted_chars'] / total_pages,
            'average_processing_speed': self.stats['total_pages'] / total_time,
            'success_rate': ((self.stats['files_processed'] - self.stats['method_usage']['failed_extractions']) / 
                            max(self.stats['files_processed'], 1)) * 100,
            'ocr_usage_rate': (self.stats['ocr_pages'] / total_pages) * 100,
            'quality_improvement_rate': (self.stats['quality_analysis']['ocr_improvements'] / 
                                        max(self.stats['files_processed'], 1)) * 100,
            
            # Library availability
            'libraries_available': self.libraries_available.copy(),
            'features_enabled': {
                'auto_method_selection': self.auto_method_selection,
                'ocr_fallback': self.enable_ocr_fallback,
                'table_extraction': self.enable_table_extraction,
                'content_validation': self.enable_content_validation,
                'structure_preservation': self.preserve_structure
            }
        }
        
        return stats
    
    def print_processing_summary(self):
        """Print comprehensive processing summary"""
        stats = self.get_processing_stats()
        
        print(f"\n📄 ENHANCED PDF PROCESSING SUMMARY:")
        print(f"   📊 Files processed: {stats['files_processed']}")
        print(f"   📄 Total pages: {stats['total_pages']}")
        print(f"   📝 Text extracted: {stats['text_extracted_chars']:,} characters")
        print(f"   ⏱️ Processing time: {stats['processing_time']:.2f}s")
        print(f"   🚀 Average speed: {stats['average_processing_speed']:.1f} pages/sec")
        print(f"   ✅ Success rate: {stats['success_rate']:.1f}%")
        
        print(f"\n🔧 Method Usage:")
        for method, count in stats['method_usage'].items():
            if count > 0:
                method_name = method.replace('_', ' ').title()
                print(f"   📋 {method_name}: {count}")
        
        print(f"\n🔄 Advanced Features:")
        print(f"   🤖 OCR pages processed: {stats['ocr_pages']} ({stats['ocr_usage_rate']:.1f}%)")
        print(f"   📊 Structured pages: {stats['structured_pages']}")
        
        # Rotation statistics
        rotation_stats = stats['rotation_stats']
        if rotation_stats['rotations_applied'] > 0:
            print(f"   🔄 Auto-rotation applied: {rotation_stats['rotations_applied']} times")
            print(f"   📈 Quality improvements: {rotation_stats['improvements_found']}")
        
        # Quality analysis
        quality_stats = stats['quality_analysis']
        total_quality_analyzed = quality_stats['high_quality_extractions'] + quality_stats['low_quality_extractions']
        if total_quality_analyzed > 0:
            high_quality_rate = (quality_stats['high_quality_extractions'] / total_quality_analyzed) * 100
            print(f"\n📊 Quality Analysis:")
            print(f"   🎯 High quality extractions: {quality_stats['high_quality_extractions']} ({high_quality_rate:.1f}%)")
            print(f"   ⚠️ Low quality extractions: {quality_stats['low_quality_extractions']}")
            print(f"   🔧 OCR improvements: {quality_stats['ocr_improvements']} ({stats['quality_improvement_rate']:.1f}%)")
        
        # Performance metrics
        if stats['average_chars_per_page'] > 0:
            print(f"\n📈 Performance Metrics:")
            print(f"   📄 Average chars per page: {stats['average_chars_per_page']:.0f}")
            if stats['files_processed'] > 0:
                avg_file_size = stats['text_extracted_chars'] / stats['files_processed']
                print(f"   📊 Average extraction per file: {avg_file_size:.0f} chars")
        
        # Feature status
        features = stats['features_enabled']
        print(f"\n⚙️ Features Status:")
        for feature, enabled in features.items():
            status = "✅" if enabled else "❌"
            feature_name = feature.replace('_', ' ').title()
            print(f"   {status} {feature_name}")
    
    def reset_stats(self):
        """Reset processing statistics"""
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
            },
            'rotation_stats': {
                'images_tested': 0,
                'rotations_applied': 0,
                'improvements_found': 0,
                'timeouts': 0
            },
            'quality_analysis': {
                'high_quality_extractions': 0,
                'low_quality_extractions': 0,
                'ocr_improvements': 0
            }
        }
        print("📊 Enhanced PDF processor statistics reset")


def create_enhanced_pdf_processor(config=None):
    """
    Create Enhanced PDF processor instance
    
    Args:
        config: Configuration object with PDF settings
    
    Returns:
        EnhancedPDFProcessor: Configured enhanced PDF processor
    """
    return EnhancedPDFProcessor(config)


def check_enhanced_pdf_capabilities():
    """
    Check available enhanced PDF processing capabilities
    
    Returns:
        dict: Comprehensive capability information
    """
    capabilities = {
        'libraries': {
            'pymupdf': PYMUPDF_AVAILABLE,
            'pdfplumber': PDFPLUMBER_AVAILABLE,
            'pdf2image': PDF2IMAGE_AVAILABLE
        },
        'features': {
            'basic_extraction': PYMUPDF_AVAILABLE or PDFPLUMBER_AVAILABLE,
            'table_extraction': PDFPLUMBER_AVAILABLE,
            'ocr_fallback': PDF2IMAGE_AVAILABLE,
            'auto_rotation': PDF2IMAGE_AVAILABLE,  # Requires OCR
            'quality_analysis': True,  # Always available
            'method_selection': PYMUPDF_AVAILABLE and PDFPLUMBER_AVAILABLE
        },
        'recommendations': []
    }
    
    # Overall status
    if capabilities['libraries']['pymupdf'] and capabilities['libraries']['pdfplumber']:
        capabilities['overall_status'] = 'excellent'
    elif any(capabilities['libraries'].values()):
        capabilities['overall_status'] = 'good'
    else:
        capabilities['overall_status'] = 'unavailable'
    
    # Generate recommendations
    if not capabilities['libraries']['pymupdf']:
        capabilities['recommendations'].append("Install PyMuPDF for fast PDF processing: pip install PyMuPDF")
    
    if not capabilities['libraries']['pdfplumber']:
        capabilities['recommendations'].append("Install pdfplumber for table extraction: pip install pdfplumber")
    
    if not capabilities['libraries']['pdf2image']:
        capabilities['recommendations'].append("Install pdf2image for OCR fallback: pip install pdf2image")
    
    if not any(capabilities['libraries'].values()):
        capabilities['recommendations'].append("Install all PDF libraries: pip install PyMuPDF pdfplumber pdf2image")
    
    # Print status
    print("📄 Enhanced PDF Processing Capabilities:")
    print(f"   PyMuPDF (speed): {'✅' if capabilities['libraries']['pymupdf'] else '❌'}")
    print(f"   pdfplumber (tables): {'✅' if capabilities['libraries']['pdfplumber'] else '❌'}")
    print(f"   pdf2image (OCR): {'✅' if capabilities['libraries']['pdf2image'] else '❌'}")
    
    print(f"\n🚀 Available Features:")
    for feature, available in capabilities['features'].items():
        status = "✅" if available else "❌"
        feature_name = feature.replace('_', ' ').title()
        print(f"   {status} {feature_name}")
    
    print(f"\n📊 Overall Status: {capabilities['overall_status'].upper()}")
    
    if capabilities['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in capabilities['recommendations']:
            print(f"   • {rec}")
    
    return capabilities


def validate_pdf_file(pdf_path):
    """
    Validate PDF file for processing
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        tuple: (is_valid, error_message, file_info)
    """
    try:
        # Basic file validation
        is_valid, error_msg = validate_file_path(pdf_path)
        if not is_valid:
            return False, error_msg, None
        
        # Get file info
        file_info = get_file_info(pdf_path)
        if 'error' in file_info:
            return False, file_info['error'], None
        
        # Check file extension
        if not pdf_path.lower().endswith('.pdf'):
            return False, "File is not a PDF", file_info
        
        # Try to open with PyMuPDF if available
        if PYMUPDF_AVAILABLE:
            try:
                import fitz
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                doc.close()
                
                if page_count == 0:
                    return False, "PDF has no pages", file_info
                
                file_info['page_count'] = page_count
                
            except Exception as e:
                return False, f"Cannot open PDF: {e}", file_info
        
        return True, None, file_info
        
    except Exception as e:
        return False, f"Validation error: {e}", None


# Convenience functions for backward compatibility
def create_pdf_processor(config=None):
    """
    Create PDF processor (alias for enhanced version)
    
    Args:
        config: Configuration object
    
    Returns:
        EnhancedPDFProcessor: Enhanced PDF processor instance
    """
    return create_enhanced_pdf_processor(config)


def check_pdf_processing_capabilities():
    """
    Check PDF processing capabilities (alias for enhanced version)
    
    Returns:
        dict: Capability information
    """
    return check_enhanced_pdf_capabilities()


if __name__ == "__main__":
    # Test enhanced PDF processor capabilities when run directly
    print("🧪 Enhanced PDF Processor - Capability Test")
    print("=" * 60)
    
    capabilities = check_enhanced_pdf_capabilities()
    
    if capabilities['overall_status'] != 'unavailable':
        print(f"\n✅ Enhanced PDF processor ready!")
        print(f"📄 Features available: {sum(capabilities['features'].values())}/{len(capabilities['features'])}")
    else:
        print(f"\n❌ Enhanced PDF processor not available")
        print(f"📋 Install required libraries to enable PDF processing")
    
    print("=" * 60)
