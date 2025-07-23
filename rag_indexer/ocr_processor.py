#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced OCR processing module for RAG Document Indexer
Handles image text extraction using Tesseract OCR with auto-rotation and text quality analysis
"""

import os
import re
import time
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


class TextQualityAnalyzer:
    """Analyzer for determining text meaningfulness and quality"""
    
    def __init__(self, language='english'):
        """
        Initialize text quality analyzer
        
        Args:
            language: Language for analysis ('english', 'russian', 'auto')
        """
        self.language = language
        
        # Common English words for quality checking
        self.english_common_words = {
            'the', 'and', 'or', 'of', 'to', 'in', 'a', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with',
            'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'not', 'word', 'but', 'what', 'some', 'we', 'can', 'out',
            'other', 'were', 'all', 'there', 'when', 'up', 'use', 'your', 'how', 'said', 'an', 'each', 'which', 'she', 'do',
            'one', 'their', 'time', 'will', 'about', 'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
            'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more', 'very', 'what', 'know', 'just', 'first', 'get', 'over'
        }
        
        # Common Russian words
        self.russian_common_words = {
            '?', '?', '??', '??', '?', '????', '?', '?', '?', '??', '??', '???', '?', '??', '???', '???', '???', '??', '??', '??',
            '??', '??', '??', '???', '??', '?', '??', '???', '???', '???', '??', '???', '????', '????', '????????', '?????', '???'
        }
        
        # Letter frequency for English (normalized)
        self.english_letter_freq = {
            'e': 0.127, 't': 0.091, 'a': 0.082, 'o': 0.075, 'i': 0.070, 'n': 0.067, 's': 0.063, 'h': 0.061, 'r': 0.060,
            'd': 0.043, 'l': 0.040, 'c': 0.028, 'u': 0.028, 'm': 0.024, 'w': 0.023, 'f': 0.022, 'g': 0.020, 'y': 0.020,
            'p': 0.019, 'b': 0.013, 'v': 0.010, 'k': 0.008, 'j': 0.001, 'x': 0.001, 'q': 0.001, 'z': 0.001
        }
        
        # Letter frequency for Russian (normalized)
        self.russian_letter_freq = {
            '?': 0.110, '?': 0.084, '?': 0.074, '?': 0.073, '?': 0.067, '?': 0.062, '?': 0.055, '?': 0.047, '?': 0.045,
            '?': 0.044, '?': 0.035, '?': 0.032, '?': 0.030, '?': 0.028, '?': 0.026, '?': 0.020, '?': 0.019, '?': 0.017,
            '?': 0.017, '?': 0.016, '?': 0.016, '?': 0.014, '?': 0.012, '?': 0.010, '?': 0.009, '?': 0.007, '?': 0.006
        }
    
    def detect_language(self, text):
        """
        Detect text language based on character patterns
        
        Args:
            text: Text to analyze
        
        Returns:
            str: Detected language ('english', 'russian', 'unknown')
        """
        if not text:
            return 'unknown'
        
        # Count Cyrillic vs Latin characters
        cyrillic_chars = len([c for c in text.lower() if '?' <= c <= '?' or c == '?'])
        latin_chars = len([c for c in text.lower() if 'a' <= c <= 'z'])
        
        total_letters = cyrillic_chars + latin_chars
        if total_letters < 10:
            return 'unknown'
        
        cyrillic_ratio = cyrillic_chars / total_letters
        
        if cyrillic_ratio > 0.7:
            return 'russian'
        elif cyrillic_ratio < 0.3:
            return 'english'
        else:
            return 'mixed'
    
    def calculate_letter_frequency_score(self, text, language):
        """
        Calculate how well text matches expected letter frequency for language
        
        Args:
            text: Text to analyze
            language: Language to check against
        
        Returns:
            float: Frequency score (0-1, higher is better)
        """
        text_lower = text.lower()
        letter_counts = {}
        
        # Count letters only
        total_letters = 0
        for char in text_lower:
            if char.isalpha():
                letter_counts[char] = letter_counts.get(char, 0) + 1
                total_letters += 1
        
        if total_letters < 20:
            return 0.5  # Not enough data
        
        # Normalize counts to frequencies
        text_freq = {char: count/total_letters for char, count in letter_counts.items()}
        
        # Get expected frequencies for language
        if language == 'russian':
            expected_freq = self.russian_letter_freq
        else:
            expected_freq = self.english_letter_freq
        
        # Calculate chi-squared-like score (simplified)
        score = 0.0
        total_expected_chars = 0
        
        for char, expected in expected_freq.items():
            if char in text_freq:
                # Penalize large deviations from expected frequency
                deviation = abs(text_freq[char] - expected)
                score += max(0, expected - deviation)
                total_expected_chars += expected
        
        return score / total_expected_chars if total_expected_chars > 0 else 0.0
    
    def calculate_word_quality_score(self, text, language):
        """
        Calculate quality based on presence of common words
        
        Args:
            text: Text to analyze
            language: Language to check
        
        Returns:
            float: Word quality score (0-1)
        """
        if not text:
            return 0.0
        
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        
        # Get common words for language
        if language == 'russian':
            common_words = self.russian_common_words
        else:
            common_words = self.english_common_words
        
        # Count common words
        common_word_count = sum(1 for word in words if word in common_words)
        
        return common_word_count / len(words)
    
    def analyze_text_structure(self, text):
        """
        Analyze text structure for meaningfulness indicators
        
        Args:
            text: Text to analyze
        
        Returns:
            dict: Structure analysis results
        """
        if not text:
            return {
                'total_chars': 0,
                'letters': 0,
                'digits': 0,
                'spaces': 0,
                'punctuation': 0,
                'words': 0,
                'avg_word_length': 0,
                'sentences': 0,
                'letter_ratio': 0,
                'word_ratio': 0,
                'has_punctuation': False,
                'has_capitalization': False,
                'repetitive_chars': 0
        }
        
        total_chars = len(text)
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        spaces = sum(c.isspace() for c in text)
        punctuation = sum(c in '.,!?;:()[]{}"\'-' for c in text)
        
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        
        sentences = len(re.findall(r'[.!?]+', text))
        
        # Check for repetitive character sequences
        repetitive_chars = 0
        for i in range(len(text) - 2):
            if text[i] == text[i+1] == text[i+2]:
                repetitive_chars += 1
        
        return {
            'total_chars': total_chars,
            'letters': letters,
            'digits': digits,
            'spaces': spaces,
            'punctuation': punctuation,
            'words': word_count,
            'avg_word_length': avg_word_length,
            'sentences': sentences,
            'letter_ratio': letters / total_chars if total_chars > 0 else 0,
            'word_ratio': word_count / (total_chars / 5) if total_chars > 0 else 0,  # Rough words per char ratio
            'has_punctuation': punctuation > 0,
            'has_capitalization': any(c.isupper() for c in text),
            'repetitive_chars': repetitive_chars
        }
    
    def calculate_quality_score(self, text, min_words=5, max_identical_chars=10):
        """
        Calculate overall text quality score combining multiple metrics
        
        Args:
            text: Text to analyze
            min_words: Minimum number of words required
            max_identical_chars: Maximum allowed repetitive characters
        
        Returns:
            tuple: (quality_score, detailed_metrics)
        """
        if not text or len(text.strip()) < 10:
            return 0.0, {'reason': 'too_short', 'length': len(text) if text else 0}
        
        # Detect language if auto
        detected_language = self.language
        if self.language == 'auto':
            detected_language = self.detect_language(text)
            if detected_language == 'unknown':
                detected_language = 'english'  # Default fallback
        
        # Analyze structure
        structure = self.analyze_text_structure(text)
        
        # Calculate component scores
        letter_freq_score = self.calculate_letter_frequency_score(text, detected_language)
        word_quality_score = self.calculate_word_quality_score(text, detected_language)
        
        # Basic quality checks
        if structure['words'] < min_words:
            return 0.0, {
                'reason': 'too_few_words',
                'words': structure['words'],
                'min_required': min_words,
                'detected_language': detected_language
            }
        
        if structure['repetitive_chars'] > max_identical_chars:
            return 0.0, {
                'reason': 'too_repetitive',
                'repetitive_chars': structure['repetitive_chars'],
                'max_allowed': max_identical_chars,
                'detected_language': detected_language
            }
        
        # Calculate weighted quality score
        scores = {
            'letter_ratio': min(structure['letter_ratio'] * 2, 1.0),  # Weight: letters should dominate
            'avg_word_length': min(max(structure['avg_word_length'] - 1, 0) / 8, 1.0),  # 2-10 chars optimal
            'word_quality': word_quality_score,  # Presence of common words
            'letter_frequency': letter_freq_score,  # Matches language patterns
            'punctuation_bonus': 0.1 if structure['has_punctuation'] else 0.0,
            'capitalization_bonus': 0.1 if structure['has_capitalization'] else 0.0
        }
        
        # Weighted average (letter_ratio and word_quality are most important)
        weights = {
            'letter_ratio': 0.3,
            'avg_word_length': 0.15,
            'word_quality': 0.3,
            'letter_frequency': 0.15,
            'punctuation_bonus': 0.05,
            'capitalization_bonus': 0.05
        }
        
        weighted_score = sum(scores[key] * weights[key] for key in scores.keys())
        
        detailed_metrics = {
            'detected_language': detected_language,
            'structure': structure,
            'component_scores': scores,
            'weighted_score': weighted_score,
            'quality_indicators': {
                'meaningful_words': word_quality_score > 0.1,
                'proper_structure': structure['letter_ratio'] > 0.6,
                'reasonable_length': structure['avg_word_length'] > 2,
                'has_sentences': structure['sentences'] > 0,
                'not_repetitive': structure['repetitive_chars'] <= max_identical_chars
            }
        }
        
        return weighted_score, detailed_metrics


class OCRProcessor:
    """Enhanced OCR processor class with auto-rotation and quality analysis"""
    
    def __init__(self, quality_threshold=0.3, batch_size=10, config=None):
        """
        Initialize enhanced OCR processor
        
        Args:
            quality_threshold: Minimum quality threshold for text extraction
            batch_size: Number of images to process in one batch  
            config: Configuration object with OCR settings
        """
        self.quality_threshold = quality_threshold
        self.batch_size = batch_size
        self.is_available = OCR_AVAILABLE
        
        # Load configuration settings
        if config:
            ocr_settings = config.get_ocr_settings()
            quality_settings = config.get_text_quality_settings()
            
            self.auto_rotation = ocr_settings.get('auto_rotation', True)
            self.rotation_quality_threshold = ocr_settings.get('rotation_quality_threshold', 0.1)
            self.test_all_rotations = ocr_settings.get('test_all_rotations', False)
            self.rotation_timeout = ocr_settings.get('rotation_timeout', 30)
            self.skip_rotation_for_good_quality = ocr_settings.get('skip_rotation_for_good_quality', True)
            
            self.text_quality_enabled = quality_settings.get('enabled', True)
            self.text_quality_min_score = quality_settings.get('min_score', 0.3)
            self.text_quality_min_words = quality_settings.get('min_words', 5)
            self.text_quality_max_identical_chars = quality_settings.get('max_identical_chars', 10)
            self.text_quality_language = quality_settings.get('language', 'english')
        else:
            # Default settings
            self.auto_rotation = True
            self.rotation_quality_threshold = 0.1
            self.test_all_rotations = False
            self.rotation_timeout = 30
            self.skip_rotation_for_good_quality = True
            
            self.text_quality_enabled = True
            self.text_quality_min_score = 0.3
            self.text_quality_min_words = 5
            self.text_quality_max_identical_chars = 10
            self.text_quality_language = 'english'
        
        # Initialize text quality analyzer
        if self.text_quality_enabled:
            self.quality_analyzer = TextQualityAnalyzer(self.text_quality_language)
        else:
            self.quality_analyzer = None
        
        # Statistics
        self.rotation_stats = {
            'images_tested': 0,
            'rotations_applied': 0,
            'improvements_found': 0,
            'timeouts': 0
        }
        
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
    
    def extract_text_simple(self, image, config_override=None):
        """
        Extract text from image using OCR without rotation testing
        
        Args:
            image: PIL Image object
            config_override: Optional OCR config override
        
        Returns:
            tuple: (text, confidence_info)
        """
        if not self.is_available:
            return "", {'error': 'OCR not available'}
        
        try:
            # Safe OCR configuration
            safe_config = config_override or r'--oem 3 --psm 6'
            
            # Extract text
            text = pytesseract.image_to_string(image, lang='eng', config=safe_config)
            
            # Clean up text
            text = clean_text_from_null_bytes(text)
            text = text.strip()
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            # Try to get confidence data if available
            confidence_info = {'method': 'simple_ocr'}
            try:
                data = pytesseract.image_to_data(image, lang='eng', config=safe_config, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                if confidences:
                    confidence_info['avg_confidence'] = sum(confidences) / len(confidences)
                    confidence_info['min_confidence'] = min(confidences)
                    confidence_info['max_confidence'] = max(confidences)
            except:
                pass  # Confidence data not critical
            
            return text, confidence_info
            
        except Exception as e:
            return "", {'error': str(e)}
    
    def test_rotation_quality(self, image, rotation_angle):
        """
        Test OCR quality for a specific rotation angle
        
        Args:
            image: PIL Image object
            rotation_angle: Angle to rotate (0, 90, 180, 270)
        
        Returns:
            tuple: (text, quality_score, rotation_info)
        """
        try:
            # Rotate image if needed
            if rotation_angle == 0:
                test_image = image
            else:
                test_image = image.rotate(-rotation_angle, expand=True)
            
            # Extract text
            text, ocr_info = self.extract_text_simple(test_image)
            
            # Calculate quality score
            if self.text_quality_enabled and self.quality_analyzer:
                quality_score, quality_metrics = self.quality_analyzer.calculate_quality_score(
                    text, 
                    self.text_quality_min_words, 
                    self.text_quality_max_identical_chars
                )
            else:
                # Simple quality score based on text characteristics
                if not text:
                    quality_score = 0.0
                else:
                    letters = sum(c.isalpha() for c in text)
                    total_chars = len(text.replace(' ', '').replace('\n', ''))
                    quality_score = letters / total_chars if total_chars > 0 else 0.0
                quality_metrics = {'simple_score': True}
            
            rotation_info = {
                'angle': rotation_angle,
                'text_length': len(text),
                'quality_score': quality_score,
                'ocr_info': ocr_info,
                'quality_metrics': quality_metrics
            }
            
            return text, quality_score, rotation_info
            
        except Exception as e:
            return "", 0.0, {'angle': rotation_angle, 'error': str(e)}
    
    def detect_best_rotation(self, image):
        """
        Detect best rotation angle for OCR by testing different angles
        
        Args:
            image: PIL Image object
        
        Returns:
            tuple: (best_text, best_angle, rotation_results)
        """
        if not self.auto_rotation:
            text, ocr_info = self.extract_text_simple(image)
            return text, 0, {'auto_rotation_disabled': True, 'ocr_info': ocr_info}
        
        start_time = time.time()
        
        # Test angles
        angles_to_test = [0, 90, 180, 270]
        results = []
        
        self.rotation_stats['images_tested'] += 1
        
        try:
            # Test each rotation angle
            for angle in angles_to_test:
                if time.time() - start_time > self.rotation_timeout:
                    self.rotation_stats['timeouts'] += 1
                    print(f"   WARNING: Rotation testing timed out after {self.rotation_timeout}s")
                    break
                
                text, quality_score, rotation_info = self.test_rotation_quality(image, angle)
                results.append((text, quality_score, rotation_info))
                
                # Early exit if we find good quality and skip_rotation_for_good_quality is enabled
                if (angle == 0 and self.skip_rotation_for_good_quality and 
                    quality_score >= self.text_quality_min_score):
                    print(f"   INFO: Original rotation has good quality ({quality_score:.2f}), skipping other rotations")
                    break
                
                # Early exit if test_all_rotations is False and we found a good result
                if not self.test_all_rotations and quality_score >= self.text_quality_min_score:
                    break
            
            if not results:
                return "", 0, {'error': 'No rotation results', 'timeout': True}
            
            # Find best result
            best_result = max(results, key=lambda x: x[1])  # Sort by quality score
            best_text, best_quality, best_info = best_result
            best_angle = best_info['angle']
            
            # Check if rotation improved quality significantly
            original_quality = results[0][1] if results else 0  # First result is always angle 0
            quality_improvement = best_quality - original_quality
            
            if best_angle != 0 and quality_improvement >= self.rotation_quality_threshold:
                self.rotation_stats['rotations_applied'] += 1
                self.rotation_stats['improvements_found'] += 1
                print(f"   INFO: Applied {best_angle}° rotation (quality: {original_quality:.2f} ? {best_quality:.2f})")
            elif best_angle != 0:
                # Rotation didn't improve enough, use original
                best_text, best_quality, best_info = results[0]
                best_angle = 0
                print(f"   INFO: Rotation improvement too small ({quality_improvement:.2f}), keeping original")
            
            rotation_results = {
                'tested_angles': len(results),
                'all_results': results,
                'best_angle': best_angle,
                'best_quality': best_quality,
                'quality_improvement': quality_improvement,
                'processing_time': time.time() - start_time,
                'timeout_occurred': time.time() - start_time > self.rotation_timeout
            }
            
            return best_text, best_angle, rotation_results
            
        except Exception as e:
            print(f"   ERROR: Rotation detection failed: {e}")
            # Fallback to simple extraction
            text, ocr_info = self.extract_text_simple(image)
            return text, 0, {'error': str(e), 'fallback': True, 'ocr_info': ocr_info}
    
    def extract_text_from_image(self, image_path, languages='eng'):
        """
        Extract text from image using OCR with auto-rotation and quality analysis
        
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
            
            # Extract text with rotation detection
            text, best_angle, rotation_info = self.detect_best_rotation(processed_image)
            
            # Final text cleaning
            text = clean_text_from_null_bytes(text)
            
            return text
            
        except Exception as e:
            print(f"Error extracting text from {image_path}: {e}")
            self._log_ocr_error(image_path, str(e))
            return ""
    
    def validate_extracted_text(self, text):
        """
        Enhanced validation of extracted text quality
        
        Args:
            text: Extracted text to validate
        
        Returns:
            tuple: (is_valid, quality_score, metrics)
        """
        if not text or len(text) < 20:
            return False, 0.0, {'reason': 'too_short', 'length': len(text)}
        
        # Use text quality analyzer if enabled
        if self.text_quality_enabled and self.quality_analyzer:
            quality_score, detailed_metrics = self.quality_analyzer.calculate_quality_score(
                text, 
                self.text_quality_min_words,
                self.text_quality_max_identical_chars
            )
            
            is_valid = quality_score >= self.text_quality_min_score
            
            return is_valid, quality_score, detailed_metrics
        else:
            # Fallback to simple validation
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
                'total_chars': total_chars,
                'simple_validation': True
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
            
            # Extract text with enhanced rotation detection
            text = self.extract_text_from_image(image_path)
            
            # Enhanced validation with quality analysis
            is_valid, quality_score, metrics = self.validate_extracted_text(text)
            
            if is_valid:
                # Clean file path from null bytes
                clean_image_path = clean_text_from_null_bytes(str(image_path))
                clean_file_name = clean_text_from_null_bytes(file_name)
                
                # Create enhanced metadata
                raw_metadata = {
                    'file_path': clean_image_path,
                    'file_name': clean_file_name,
                    'file_type': 'image',
                    'file_size': file_size,
                    'ocr_extracted': True,
                    'text_length': len(text),
                    'quality_score': quality_score,
                    'ocr_metrics': metrics,
                    'ocr_enhanced_features': {
                        'auto_rotation_enabled': self.auto_rotation,
                        'text_quality_analysis': self.text_quality_enabled,
                        'language_detection': self.text_quality_language
                    }
                }
                
                # Add rotation info if available
                if hasattr(self, '_last_rotation_info'):
                    raw_metadata['rotation_info'] = self._last_rotation_info
                
                # Clean metadata recursively
                cleaned_metadata = clean_metadata_recursive(raw_metadata)
                
                # Create document with cleaned data
                doc = Document(
                    text=text,  # Already cleaned in extract_text_from_image
                    metadata=cleaned_metadata
                )
                
                print(f"  SUCCESS: Extracted {len(text)} characters (quality: {quality_score:.2f})")
                
                # Log quality details if enabled
                if self.text_quality_enabled and 'detected_language' in metrics:
                    detected_lang = metrics.get('detected_language', 'unknown')
                    print(f"  INFO: Detected language: {detected_lang}")
                
                return doc
            else:
                reason = metrics.get('reason', 'low_quality')
                print(f"  WARNING: Low quality text ({reason}, score: {quality_score:.2f})")
                
                # Log detailed failure reason
                if 'quality_indicators' in metrics:
                    indicators = metrics['quality_indicators']
                    failed_checks = [k for k, v in indicators.items() if not v]
                    if failed_checks:
                        print(f"  DETAILS: Failed checks: {', '.join(failed_checks)}")
                
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
        Process all images in directory and extract text with enhanced features
        
        Args:
            directory: Directory containing images
        
        Returns:
            tuple: (documents, stats) where documents is list of Document objects
                   and stats is processing statistics
        """
        if not self.is_available:
            print("OCR not available. Skipping image processing.")
            return [], {'processed': 0, 'successful': 0, 'error': 'OCR not available'}
        
        print("Scanning for images with enhanced OCR processing...")
        
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
            'quality_scores': [],
            'rotation_stats': self.rotation_stats.copy(),
            'language_detection': {},
            'quality_failures': {}
        }
        
        print(f"Found {len(image_files)} image files to process")
        print(f"Enhanced features: Auto-rotation: {'?' if self.auto_rotation else '?'}, "
              f"Quality analysis: {'?' if self.text_quality_enabled else '?'}")
        
        # Process images
        for image_path in image_files:
            stats['processed'] += 1
            
            doc = self.process_single_image(image_path)
            
            if doc is not None:
                documents.append(doc)
                stats['successful'] += 1
                stats['total_text_length'] += len(doc.text)
                stats['quality_scores'].append(doc.metadata['quality_score'])
                
                # Track language detection stats
                if 'ocr_metrics' in doc.metadata and 'detected_language' in doc.metadata['ocr_metrics']:
                    lang = doc.metadata['ocr_metrics']['detected_language']
                    stats['language_detection'][lang] = stats['language_detection'].get(lang, 0) + 1
                
            else:
                stats['failed'] += 1
                
                # Try to get failure reason for stats
                try:
                    # Re-extract to get failure details (only for stats)
                    test_text = self.extract_text_from_image(image_path)
                    if test_text:
                        _, _, metrics = self.validate_extracted_text(test_text)
                        failure_reason = metrics.get('reason', 'unknown')
                        stats['quality_failures'][failure_reason] = stats['quality_failures'].get(failure_reason, 0) + 1
                except:
                    stats['quality_failures']['extraction_error'] = stats['quality_failures'].get('extraction_error', 0) + 1
            
            # Progress update
            if stats['processed'] % 10 == 0:
                print(f"  Progress: {stats['processed']}/{len(image_files)} images processed")
        
        # Update rotation stats from processor
        stats['rotation_stats'] = self.rotation_stats.copy()
        
        # Calculate final statistics
        if stats['quality_scores']:
            stats['average_quality'] = sum(stats['quality_scores']) / len(stats['quality_scores'])
            stats['min_quality'] = min(stats['quality_scores'])
            stats['max_quality'] = max(stats['quality_scores'])
        else:
            stats['average_quality'] = 0.0
            stats['min_quality'] = 0.0
            stats['max_quality'] = 0.0
        
        stats['success_rate'] = (stats['successful'] / stats['processed'] * 100) if stats['processed'] > 0 else 0
        
        # Print enhanced summary
        print(f"\n?? Enhanced Image Processing Complete:")
        print(f"  ?? Images found: {stats['total_found']}")
        print(f"  ??  Images processed: {stats['processed']}")
        print(f"  ? Successful extractions: {stats['successful']}")
        print(f"  ? Failed extractions: {stats['failed']}")
        print(f"  ?? Success rate: {stats['success_rate']:.1f}%")
        print(f"  ?? Total text extracted: {stats['total_text_length']} characters")
        
        if stats['average_quality'] > 0:
            print(f"  ?? Average quality score: {stats['average_quality']:.2f}")
            print(f"  ?? Quality range: {stats['min_quality']:.2f} - {stats['max_quality']:.2f}")
        
        # Print rotation statistics
        if self.auto_rotation:
            rotation_stats = stats['rotation_stats']
            print(f"  ?? Rotation analysis:")
            print(f"    - Images tested: {rotation_stats['images_tested']}")
            print(f"    - Rotations applied: {rotation_stats['rotations_applied']}")
            print(f"    - Improvements found: {rotation_stats['improvements_found']}")
            if rotation_stats['timeouts'] > 0:
                print(f"    - Timeouts: {rotation_stats['timeouts']}")
        
        # Print language detection stats
        if stats['language_detection']:
            print(f"  ?? Language detection:")
            for lang, count in stats['language_detection'].items():
                print(f"    - {lang}: {count} images")
        
        # Print failure analysis
        if stats['quality_failures']:
            print(f"  ??  Quality failure reasons:")
            for reason, count in stats['quality_failures'].items():
                print(f"    - {reason}: {count} images")
        
        return documents, stats
    
    def get_processing_stats(self):
        """
        Get comprehensive processing statistics
        
        Returns:
            dict: Processing statistics including rotation and quality metrics
        """
        return {
            'rotation_stats': self.rotation_stats.copy(),
            'settings': {
                'auto_rotation': self.auto_rotation,
                'text_quality_enabled': self.text_quality_enabled,
                'quality_threshold': self.quality_threshold,
                'text_quality_min_score': self.text_quality_min_score,
                'rotation_quality_threshold': self.rotation_quality_threshold,
                'language': self.text_quality_language
            }
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.rotation_stats = {
            'images_tested': 0,
            'rotations_applied': 0,
            'improvements_found': 0,
            'timeouts': 0
        }


def create_ocr_processor(quality_threshold=0.3, batch_size=10, config=None):
    """
    Create an enhanced OCR processor instance
    
    Args:
        quality_threshold: Minimum quality threshold for text extraction
        batch_size: Number of images to process in one batch
        config: Configuration object with enhanced OCR settings
    
    Returns:
        OCRProcessor: Configured enhanced OCR processor
    """
    return OCRProcessor(
        quality_threshold=quality_threshold, 
        batch_size=batch_size,
        config=config
    )


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


def test_text_quality_analyzer():
    """Test function for the text quality analyzer"""
    analyzer = TextQualityAnalyzer('auto')
    
    test_texts = [
        "This is a normal English sentence with proper structure.",
        "aaaaaaaaaaaaaaaaaaaaaaaaa",  # Repetitive
        "abc xyz 123 !@#",  # Low quality
        "??? ?????????? ??????? ????? ? ?????????? ??????????.",  # Russian
        "The quick brown fox jumps over the lazy dog."  # High quality English
    ]
    
    print("?? Testing Text Quality Analyzer:")
    for i, text in enumerate(test_texts, 1):
        score, metrics = analyzer.calculate_quality_score(text)
        lang = metrics.get('detected_language', 'unknown')
        print(f"  {i}. Score: {score:.2f}, Lang: {lang}, Text: {text[:50]}...")
        
        if 'reason' in metrics:
            print(f"     Reason: {metrics['reason']}")
    print()