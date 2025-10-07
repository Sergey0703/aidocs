#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document converter module using Docling 2.x
Compatible with Docling 2.55.1+ API
Automatic backend detection (pypdfium2)
"""

import time
from pathlib import Path
from datetime import datetime

# Docling 2.x imports
from docling.document_converter import DocumentConverter as DoclingConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from .metadata_extractor import MetadataExtractor
from .utils_docling import safe_write_file, format_time


class DocumentConverter:
    """Converter for documents using Docling 2.x"""
    
    def __init__(self, config):
        """
        Initialize document converter.
        
        Args:
            config: DoclingConfig instance.
        """
        self.config = config
        self.metadata_extractor = MetadataExtractor(config)
        self.docling = self._init_docling_converter()
        self.stats = {
            'total_files': 0, 'successful': 0, 'failed': 0, 'total_time': 0,
            'failed_files': [], 'total_batch_time': 0
        }
        
        # Print Docling version info
        self._print_docling_info()
    
    def _print_docling_info(self):
        """Print Docling version and configuration info"""
        try:
            import docling
            version = getattr(docling, '__version__', 'unknown')
            print(f"📦 Docling version: {version}")
            
            # Check if pypdfium2 is available
            try:
                import pypdfium2
                pypdfium2_version = getattr(pypdfium2, '__version__', 'unknown')
                print(f"📦 pypdfium2 backend: {pypdfium2_version}")
            except ImportError:
                print("⚠️  pypdfium2 not installed - PDF processing may be limited")
        except Exception as e:
            print(f"⚠️  Could not determine Docling version: {e}")
    
    def _init_docling_converter(self):
        """
        Initialize Docling document converter for version 2.x
        
        Docling 2.x automatically detects and uses available backends:
        - PDF: pypdfium2 (automatically detected if installed)
        - DOCX: python-docx
        - PPTX: python-pptx
        - Images: PIL/Pillow + EasyOCR
        
        No manual backend configuration needed in 2.x!
        """
        print("🔧 Initializing Docling 2.x converter...")
        
        # Configure PDF pipeline options (Docling 2.x API)
        pipeline_options = PdfPipelineOptions()
        
        # Configure OCR settings
        pipeline_options.do_ocr = self.config.ENABLE_OCR
        
        # Configure table extraction
        pipeline_options.do_table_structure = self.config.EXTRACT_TABLES
        
        # Additional Docling 2.x options
        # Enable images in markdown export (if supported)
        pipeline_options.images_scale = 1.0  # Image scaling factor
        pipeline_options.generate_page_images = False  # Don't generate page screenshots
        
        # Create converter with automatic backend detection
        # Docling 2.x will automatically discover and use pypdfium2 if available
        try:
            converter = DoclingConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.DOCX,
                    InputFormat.PPTX,
                    InputFormat.HTML,
                    InputFormat.IMAGE,
                ],
                format_options={
                    InputFormat.PDF: pipeline_options,
                }
            )
            
            print("✅ Docling 2.x converter initialized successfully")
            print(f"   OCR: {'enabled' if self.config.ENABLE_OCR else 'disabled'}")
            print(f"   Table extraction: {'enabled' if self.config.EXTRACT_TABLES else 'disabled'}")
            
            return converter
            
        except Exception as e:
            print(f"❌ Failed to initialize Docling converter: {e}")
            print("   Make sure all required dependencies are installed:")
            print("   - pypdfium2 (for PDF)")
            print("   - python-docx (for DOCX)")
            print("   - python-pptx (for PPTX)")
            print("   - easyocr (for OCR)")
            raise
    
    def convert_file(self, input_path):
        """
        Convert a single file to markdown using Docling 2.x
        
        Args:
            input_path: Path to input file
            
        Returns:
            tuple: (success: bool, output_path: Path, error_msg: str)
        """
        input_path = Path(input_path)
        
        timestamp = datetime.now().strftime(self.config.TIMESTAMP_FORMAT)
        output_path = self.config.get_output_path(input_path, timestamp)
        
        print(f"\n📄 Converting: {input_path.name}")
        print(f"   → {output_path.relative_to(self.config.MARKDOWN_OUTPUT_DIR)}")
        
        start_time = time.time()
        
        try:
            # Convert document using Docling 2.x
            # The convert() method returns ConversionResult
            result = self.docling.convert(str(input_path))
            
            # Export to markdown
            # Docling 2.x: result.document.export_to_markdown()
            markdown_content = result.document.export_to_markdown()
            
            # Validate content
            if not markdown_content or len(markdown_content.strip()) < 10:
                raise ValueError("Conversion produced empty or invalid content")
            
            # Write markdown file
            if not safe_write_file(output_path, markdown_content):
                raise IOError(f"Failed to write markdown file to {output_path}")
            
            # Calculate conversion time
            conversion_time = time.time() - start_time
            
            # Extract and save metadata
            metadata = self.metadata_extractor.extract_metadata(
                input_path=input_path, 
                output_path=output_path,
                markdown_content=markdown_content, 
                conversion_time=conversion_time,
                docling_result=result
            )
            self.metadata_extractor.save_metadata(input_path, metadata)
            
            # Success message
            print(f"   ✅ Success ({conversion_time:.2f}s)")
            print(f"   📊 Size: {len(markdown_content):,} chars")
            
            return True, output_path, None
            
        except Exception as e:
            error_msg = str(e)
            conversion_time = time.time() - start_time
            
            print(f"   ❌ Failed ({conversion_time:.2f}s): {error_msg}")
            
            # Save detailed error log
            self._save_failed_conversion_log(input_path, error_msg)
            
            return False, None, error_msg
    
    def convert_batch(self, files_to_process):
        """
        Convert a batch of files, updating stats along the way.
        
        Args:
            files_to_process: List of file paths to convert
            
        Returns:
            dict: Conversion statistics
        """
        if not files_to_process:
            print("⚠️ No files to convert in this batch.")
            return self.get_conversion_stats()
        
        print(f"\n🚀 Starting conversion of {len(files_to_process)} files...")
        batch_start_time = time.time()
        
        successful_in_batch = 0
        failed_in_batch = 0
        total_time_in_batch = 0

        for i, file_path in enumerate(files_to_process, 1):
            self.stats['total_files'] += 1
            print(f"\n[{i}/{len(files_to_process)}]", end=" ")

            file_start_time = time.time()
            success, _, error_msg = self.convert_file(file_path)
            file_conversion_time = time.time() - file_start_time

            if success:
                successful_in_batch += 1
                total_time_in_batch += file_conversion_time
            else:
                failed_in_batch += 1
                self.stats['failed_files'].append({
                    'file': str(file_path), 
                    'error': error_msg, 
                    'timestamp': datetime.now().isoformat()
                })
            
            # Print progress every 5 files
            if i % 5 == 0:
                self._print_progress(i, len(files_to_process), batch_start_time)
        
        # Update cumulative stats
        self.stats['successful'] += successful_in_batch
        self.stats['failed'] += failed_in_batch
        self.stats['total_time'] += total_time_in_batch
        self.stats['total_batch_time'] = time.time() - batch_start_time
        
        # Print final summary
        self._print_final_summary()
        
        return self.get_conversion_stats()
    
    def _save_failed_conversion_log(self, input_path, error_msg):
        """
        Save information about a failed conversion to a log file.
        
        Args:
            input_path: Path to failed input file
            error_msg: Error message
        """
        try:
            failed_dir = Path(self.config.FAILED_CONVERSIONS_DIR)
            failed_dir.mkdir(parents=True, exist_ok=True)
            
            error_log_path = failed_dir / f"{input_path.stem}.error.txt"
            
            # Get Docling version for debugging
            try:
                import docling
                docling_version = docling.__version__
            except:
                docling_version = "unknown"
            
            error_info = (
                f"Docling Conversion Error\n"
                f"{'=' * 60}\n"
                f"File: {input_path}\n"
                f"Error: {error_msg}\n"
                f"Timestamp: {datetime.now().isoformat()}\n"
                f"Docling Version: {docling_version}\n"
                f"{'=' * 60}\n"
            )
            
            safe_write_file(error_log_path, error_info)
            
        except Exception as e:
            print(f"   ⚠️ Could not save failed conversion log: {e}")
    
    def _print_progress(self, current, total, start_time):
        """
        Print conversion progress.
        
        Args:
            current: Current file number
            total: Total files
            start_time: Batch start time
        """
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        
        print(f"\n📊 Progress: {current}/{total} files")
        print(f"   ✅ Successful: {self.stats['successful']}")
        print(f"   ❌ Failed: {self.stats['failed']}")
        print(f"   ⚡ Rate: {rate:.2f} files/sec")
        print(f"   ⏱️ ETA: {format_time(eta)}")
    
    def _print_final_summary(self):
        """Print final conversion summary."""
        print(f"\n" + "=" * 60)
        print(f"✅ BATCH CONVERSION COMPLETED")
        print(f"=" * 60)
        print(f"📊 Results for this batch:")
        print(f"   Total files attempted: {self.stats['total_files']}")
        print(f"   ✅ Successful: {self.stats['successful']}")
        print(f"   ❌ Failed: {self.stats['failed']}")
        
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_files']) * 100
            print(f"   📈 Success rate: {success_rate:.1f}%")
        
        if self.stats.get('total_batch_time'):
            print(f"\n⏱️ Performance for this batch:")
            print(f"   Total time: {format_time(self.stats['total_batch_time'])}")
            if self.stats['successful'] > 0:
                avg_time = self.stats['total_time'] / self.stats['successful']
                print(f"   Average per successful file: {avg_time:.2f}s")
        
        if self.stats['failed_files']:
            print(f"\n❌ Failed files (check logs in '{self.config.FAILED_CONVERSIONS_DIR}'):")
            for failed in self.stats['failed_files'][:5]:
                print(f"   - {Path(failed['file']).name}: {failed['error'][:100]}...")
            if len(self.stats['failed_files']) > 5:
                print(f"   ... and {len(self.stats['failed_files']) - 5} more.")
        
        print(f"=" * 60)
    
    def get_conversion_stats(self):
        """
        Get current conversion statistics.
        
        Returns:
            dict: Copy of statistics dictionary
        """
        return self.stats.copy()


def create_document_converter(config):
    """
    Factory function to create a DocumentConverter instance.
    
    Args:
        config: DoclingConfig instance
        
    Returns:
        DocumentConverter: Initialized converter
    """
    return DocumentConverter(config)