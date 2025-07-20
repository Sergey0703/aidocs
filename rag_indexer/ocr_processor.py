#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR processing module for RAG Document Indexer
Handles image text extraction using Tesseract OCR
"""

import os
from pathlib import Path
from datetime import datetime
from llama_index.core import Document

# --- OCR IMPORTS ---
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("WARNING: OCR libraries not installed. Run: pip install pytesseract pillow opencv-python")


def clean_text_from_null_bytes(text):
    """
    Clean text from null bytes and other problematic characters
    
    Args:
        text: Text to clean
    
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return text
    
    # Remove null bytes (\u0000) and other problematic characters
    text = text.replace('\u0000', '').replace('\x00', '')
    
    # Remove other control characters except newlines and tabs
    cleaned_text = ''.join(char for char in text 
                          if ord(char) >= 32 or char in '\n\t\r')
    
    return cleaned_text


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


class OCRProcessor:
    """OCR processor class for extracting text from images"""
    
    def __init__(self, quality_threshold=0.3, batch_size=10):
        """
        Initialize OCR processor
        
        Args:
            quality_threshold: Minimum quality threshold for text extraction
            batch_size: Number of images to process in one batch
        """
        self.quality_threshold = quality_threshold
        self.batch_size = batch_size
        self.is_available = OCR_AVAILABLE
        
        if not self.is_available:
            print("WARNING: OCR not available - image processing will be skipped")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for better OCR quality
        
        Args:
            image_path: Path to the image file
        
        Returns:
            PIL.Image: Preprocessed image or None if failed
        """
        if not self.is_available:
            return None
        
        try:
            # Open and convert image
            image = Image.open(image_path)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize small images
            width, height = image.size
            if width < 1000 or height < 1000:
                scale_factor = max(1000/width, 1000/height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Convert to grayscale and apply filter
            image = image.convert('L')
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def extract_text_from_image(self, image_path, languages='eng'):
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to the image file
            languages: OCR languages (default: 'eng')
        
        Returns:
            str: Extracted text or empty string if failed
        """
        if not self.is_available:
            return ""
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return ""
            
            # ???????????? ???????????? - ??? ?????????? ???????
            safe_config = '--oem 3 --psm 6'
            
            # Extract text
            text = pytesseract.image_to_string(
                processed_image, 
                lang=languages, 
                config=safe_config
            )
            
            # Clean up text - ?????: ??????? null bytes!
            text = clean_text_from_null_bytes(text)
            text = text.strip()
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            return text
            
        except Exception as e:
            print(f"Error extracting text from {image_path}: {e}")
            return ""
    
    def validate_extracted_text(self, text):
        """
        Validate quality of extracted text
        
        Args:
            text: Extracted text to validate
        
        Returns:
            tuple: (is_valid, quality_score, metrics)
        """
        if not text or len(text) < 20:
            return False, 0.0, {'reason': 'too_short', 'length': len(text)}
        
        # Calculate text quality metrics
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        spaces = sum(c.isspace() for c in text)
        special_chars = len(text) - letters - digits - spaces
        
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        if total_chars == 0:
            return False, 0.0, {'reason': 'no_content'}
        
        # Quality score based on letter ratio
        letter_ratio = letters / total_chars if total_chars > 0 else 0
        
        metrics = {
            'length': len(text),
            'letters': letters,
            'digits': digits,
            'spaces': spaces,
            'special_chars': special_chars,
            'letter_ratio': letter_ratio,
            'total_chars': total_chars
        }
        
        is_valid = letter_ratio > self.quality_threshold
        
        return is_valid, letter_ratio, metrics
    
    def process_single_image(self, image_path):
        """
        Process a single image and return Document if successful
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Document or None: Document object with extracted text or None if failed
        """
        if not self.is_available:
            return None
        
        try:
            # Get file info
            file_name = os.path.basename(image_path)
            file_size = os.path.getsize(image_path)
            
            print(f"Processing image: {file_name}")
            
            # Extract text with improved error handling
            text = self.extract_text_from_image(image_path)
            
            # Validate text quality
            is_valid, quality_score, metrics = self.validate_extracted_text(text)
            
            if is_valid:
                # Clean file path from null bytes
                clean_image_path = clean_text_from_null_bytes(str(image_path))
                clean_file_name = clean_text_from_null_bytes(file_name)
                
                # Create metadata and clean it
                raw_metadata = {
                    'file_path': clean_image_path,
                    'file_name': clean_file_name,
                    'file_type': 'image',
                    'file_size': file_size,
                    'ocr_extracted': True,
                    'text_length': len(text),
                    'quality_score': quality_score,
                    'ocr_metrics': metrics
                }
                
                # Clean metadata recursively
                cleaned_metadata = clean_metadata_recursive(raw_metadata)
                
                # Create document with cleaned data
                doc = Document(
                    text=text,  # Already cleaned in extract_text_from_image
                    metadata=cleaned_metadata
                )
                
                print(f"  SUCCESS: Extracted {len(text)} characters (quality: {quality_score:.2f})")
                return doc
            else:
                reason = metrics.get('reason', 'low_quality')
                print(f"  WARNING: Low quality text ({reason}, score: {quality_score:.2f})")
                return None
                
        except Exception as e:
            print(f"  ERROR: Failed to process {image_path}: {e}")
            # Log the specific error for debugging
            self._log_ocr_error(image_path, str(e))
            return None
    
    def _log_ocr_error(self, image_path, error_message):
        """Log OCR errors to file for debugging"""
        try:
            # Clean error message and path from null bytes
            clean_path = clean_text_from_null_bytes(str(image_path))
            clean_error = clean_text_from_null_bytes(str(error_message))
            
            with open('./ocr_errors.log', 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} - OCR Error: {os.path.basename(clean_path)} - {clean_error}\n")
        except:
            pass  # Silently fail if can't write log
    
    def get_image_files(self, directory):
        """
        Get list of image files in directory
        
        Args:
            directory: Directory to scan
        
        Returns:
            list: List of image file paths
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        image_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        return image_files
    
    def process_images_in_directory(self, directory):
        """
        Process all images in directory and extract text
        
        Args:
            directory: Directory containing images
        
        Returns:
            tuple: (documents, stats) where documents is list of Document objects
                   and stats is processing statistics
        """
        if not self.is_available:
            print("OCR not available. Skipping image processing.")
            return [], {'processed': 0, 'successful': 0, 'error': 'OCR not available'}
        
        print("Scanning for images...")
        
        # Get all image files
        image_files = self.get_image_files(directory)
        
        if not image_files:
            print("No image files found.")
            return [], {'processed': 0, 'successful': 0, 'message': 'No images found'}
        
        documents = []
        stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'total_found': len(image_files),
            'total_text_length': 0,
            'quality_scores': []
        }
        
        print(f"Found {len(image_files)} image files to process")
        
        # Process images
        for image_path in image_files:
            stats['processed'] += 1
            
            doc = self.process_single_image(image_path)
            
            if doc is not None:
                documents.append(doc)
                stats['successful'] += 1
                stats['total_text_length'] += len(doc.text)
                stats['quality_scores'].append(doc.metadata['quality_score'])
            else:
                stats['failed'] += 1
            
            # Progress update
            if stats['processed'] % 10 == 0:
                print(f"  Progress: {stats['processed']}/{len(image_files)} images processed")
        
        # Calculate final statistics
        if stats['quality_scores']:
            stats['average_quality'] = sum(stats['quality_scores']) / len(stats['quality_scores'])
        else:
            stats['average_quality'] = 0.0
        
        stats['success_rate'] = (stats['successful'] / stats['processed'] * 100) if stats['processed'] > 0 else 0
        
        # Print summary
        print(f"\nImage processing complete:")
        print(f"  Images found: {stats['total_found']}")
        print(f"  Images processed: {stats['processed']}")
        print(f"  Successful extractions: {stats['successful']}")
        print(f"  Failed extractions: {stats['failed']}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        print(f"  Total text extracted: {stats['total_text_length']} characters")
        if stats['average_quality'] > 0:
            print(f"  Average quality score: {stats['average_quality']:.2f}")
        
        return documents, stats


def create_ocr_processor(quality_threshold=0.3, batch_size=10):
    """
    Create an OCR processor instance
    
    Args:
        quality_threshold: Minimum quality threshold for text extraction
        batch_size: Number of images to process in one batch
    
    Returns:
        OCRProcessor: Configured OCR processor
    """
    return OCRProcessor(quality_threshold=quality_threshold, batch_size=batch_size)


def check_ocr_availability():
    """
    Check if OCR libraries are available
    
    Returns:
        tuple: (is_available, missing_libraries)
    """
    missing = []
    
    try:
        import pytesseract
    except ImportError:
        missing.append('pytesseract')
    
    try:
        from PIL import Image
    except ImportError:
        missing.append('pillow')
    
    try:
        import cv2
    except ImportError:
        missing.append('opencv-python')
    
    return len(missing) == 0, missing