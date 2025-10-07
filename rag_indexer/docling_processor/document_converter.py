#!/usr/-bin/env python3
# -*- coding: utf-8 -*-
"""
Document converter module using Docling
Converts raw documents to markdown format
"""

import time
import shutil
from pathlib import Path
from datetime import datetime
from docling.document_converter import DocumentConverter as DoclingConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from .metadata_extractor import MetadataExtractor
from .utils_docling import ensure_directory_exists, format_time, safe_write_file


class DocumentConverter:
    """Converter for documents using Docling"""
    
    def __init__(self, config):
        """
        Initialize document converter
        
        Args:
            config: DoclingConfig instance
        """
        self.config = config
        self.metadata_extractor = MetadataExtractor(config)
        
        # Initialize Docling converter
        self.docling = self._init_docling_converter()
        
        # Conversion statistics
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_time': 0,
            'failed_files': []
        }
    
    def _init_docling_converter(self):
        """
        Initialize Docling document converter
        
        Returns:
            DoclingConverter: Configured Docling converter
        """
        print("🔧 Initializing Docling converter...")
        
        # Configure PDF pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.ENABLE_OCR
        pipeline_options.do_table_structure = self.config.EXTRACT_TABLES
        
        # --- ИСПРАВЛЕНИЕ: Явно указываем "движок" для обработки PDF ---
        pipeline_options.backend = PyPdfiumDocumentBackend
        
        # Create converter
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
        
        print("✅ Docling converter initialized")
        return converter
    
    def convert_file(self, input_path):
        """
        Convert a single file to markdown
        
        Args:
            input_path: Path to input file
        
        Returns:
            tuple: (success, output_path, error_message)
        """
        input_path = Path(input_path)
        self.stats['total_files'] += 1
        
        timestamp = datetime.now().strftime(self.config.TIMESTAMP_FORMAT)
        output_path = self.config.get_output_path(input_path, timestamp)
        
        print(f"\n📄 Converting: {input_path.name}")
        print(f"   → {output_path.relative_to(self.config.MARKDOWN_OUTPUT_DIR)}")
        
        start_time = time.time()
        
        try:
            # Convert document using Docling
            result = self.docling.convert(str(input_path))
            
            # Export to markdown
            markdown_content = result.document.export_to_markdown()
            
            # Write markdown file
            if not safe_write_file(output_path, markdown_content):
                raise Exception("Failed to write markdown file")
            
            # Extract and save metadata
            conversion_time = time.time() - start_time
            metadata = self.metadata_extractor.extract_metadata(
                input_path=input_path,
                output_path=output_path,
                markdown_content=markdown_content,
                conversion_time=conversion_time,
                docling_result=result
            )
            
            self.metadata_extractor.save_metadata(input_path, metadata)
            
            # Update stats
            self.stats['successful'] += 1
            self.stats['total_time'] += conversion_time
            
            print(f"   ✅ Success ({conversion_time:.2f}s)")
            print(f"   📊 Size: {len(markdown_content):,} chars")
            
            return True, output_path, None
            
        except Exception as e:
            error_msg = str(e)
            
            self.stats['failed'] += 1
            self.stats['failed_files'].append({
                'file': str(input_path),
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"   ❌ Failed: {error_msg}")
            
            if self.config.SKIP_FAILED_CONVERSIONS:
                self._save_failed_conversion_log(input_path, error_msg)
            
            return False, None, error_msg
    
    def convert_batch(self, files_to_process):
        """
        Convert a batch of files
        
        Args:
            files_to_process: List of file paths
        
        Returns:
            dict: Conversion results
        """
        if not files_to_process:
            print("⚠️ No files to convert")
            return self.get_conversion_stats()
        
        print(f"\n🚀 Starting conversion of {len(files_to_process)} files...")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print("=" * 60)
        
        batch_start = time.time()
        
        for i, file_path in enumerate(files_to_process, 1):
            print(f"\n[{i}/{len(files_to_process)}]", end=" ")
            self.convert_file(file_path)
            
            if i % 5 == 0:
                self._print_progress(i, len(files_to_process), batch_start)
        
        self.stats['total_batch_time'] = time.time() - batch_start
        self._print_final_summary()
        
        return self.get_conversion_stats()
    
    def _save_failed_conversion_log(self, input_path, error_msg):
        """
        Save information about a failed conversion to a log file.
        """
        try:
            failed_dir = Path(self.config.FAILED_CONVERSIONS_DIR)
            failed_dir.mkdir(parents=True, exist_ok=True)
            
            error_log_path = failed_dir / f"{input_path.stem}.error.txt"
            error_info = f"File: {input_path}\nError: {error_msg}\nTimestamp: {datetime.now().isoformat()}\n"
            
            safe_write_file(error_log_path, error_info)
            
        except Exception as e:
            print(f"   ⚠️ Could not save failed conversion log: {e}")
    
    def _print_progress(self, current, total, start_time):
        """Print conversion progress"""
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        
        print(f"\n📊 Progress: {current}/{total} files")
        print(f"   ✅ Successful: {self.stats['successful']}")
        print(f"   ❌ Failed: {self.stats['failed']}")
        print(f"   ⚡ Rate: {rate:.2f} files/sec")
        print(f"   ⏱️ ETA: {format_time(eta)}")
    
    def _print_final_summary(self):
        """Print final conversion summary"""
        print(f"\n" + "=" * 60)
        print(f"✅ CONVERSION COMPLETED")
        print(f"=" * 60)
        print(f"📊 Results:")
        print(f"   Total files attempted: {self.stats['total_files']}")
        print(f"   ✅ Successful: {self.stats['successful']}")
        print(f"   ❌ Failed: {self.stats['failed']}")
        
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_files']) * 100
            print(f"   📈 Success rate: {success_rate:.1f}%")
        
        total_batch_time = self.stats.get('total_batch_time')
        if total_batch_time:
            print(f"\n⏱️ Performance:")
            print(f"   Total time: {format_time(total_batch_time)}")
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
        """Get conversion statistics"""
        return self.stats.copy()


def create_document_converter(config):
    """
    Create document converter instance
    
    Args:
        config: DoclingConfig instance
    
    Returns:
        DocumentConverter: Converter instance
    """
    return DocumentConverter(config)