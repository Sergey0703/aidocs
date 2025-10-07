#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document converter module using Docling.
This version relies on Docling's automatic backend detection.
"""

import time
from pathlib import Path
from datetime import datetime
from docling.document_converter import DocumentConverter as DoclingConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from .metadata_extractor import MetadataExtractor
from .utils_docling import safe_write_file, format_time

class DocumentConverter:
    """Converter for documents using Docling"""
    
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
    
    def _init_docling_converter(self):
        """
        Initialize Docling document converter.
        """
        print("🔧 Initializing Docling converter...")
        
        # Configure options for the PDF pipeline (e.g., OCR).
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.ENABLE_OCR
        pipeline_options.do_table_structure = self.config.EXTRACT_TABLES
        
        # Create the converter. Docling will automatically discover and use
        # the installed PDF backend (pypdfium2) if it's available.
        # No manual backend configuration is needed.
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
        Convert a single file to markdown.
        """
        input_path = Path(input_path)
        
        timestamp = datetime.now().strftime(self.config.TIMESTAMP_FORMAT)
        output_path = self.config.get_output_path(input_path, timestamp)
        
        print(f"\n📄 Converting: {input_path.name}")
        print(f"   → {output_path.relative_to(self.config.MARKDOWN_OUTPUT_DIR)}")
        
        start_time = time.time()
        
        try:
            result = self.docling.convert(str(input_path))
            markdown_content = result.document.export_to_markdown()
            
            if not safe_write_file(output_path, markdown_content):
                raise IOError(f"Failed to write markdown file to {output_path}")
            
            conversion_time = time.time() - start_time
            metadata = self.metadata_extractor.extract_metadata(
                input_path=input_path, output_path=output_path,
                markdown_content=markdown_content, conversion_time=conversion_time,
                docling_result=result
            )
            self.metadata_extractor.save_metadata(input_path, metadata)
            
            print(f"   ✅ Success ({conversion_time:.2f}s)")
            print(f"   📊 Size: {len(markdown_content):,} chars")
            
            return True, output_path, None
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Failed: {error_msg}")
            self._save_failed_conversion_log(input_path, error_msg)
            return False, None, error_msg

    def convert_batch(self, files_to_process):
        """
        Convert a batch of files, updating stats along the way.
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
                    'file': str(file_path), 'error': error_msg, 'timestamp': datetime.now().isoformat()
                })
            
            if i % 5 == 0:
                self._print_progress(i, len(files_to_process), batch_start_time)
        
        self.stats['successful'] += successful_in_batch
        self.stats['failed'] += failed_in_batch
        self.stats['total_time'] += total_time_in_batch
        self.stats['total_batch_time'] = time.time() - batch_start_time
        
        self._print_final_summary()
        return self.get_conversion_stats()
    
    def _save_failed_conversion_log(self, input_path, error_msg):
        """Save information about a failed conversion to a log file."""
        try:
            failed_dir = Path(self.config.FAILED_CONVERSIONS_DIR)
            failed_dir.mkdir(parents=True, exist_ok=True)
            error_log_path = failed_dir / f"{input_path.stem}.error.txt"
            error_info = f"File: {input_path}\nError: {error_msg}\nTimestamp: {datetime.now().isoformat()}\n"
            safe_write_file(error_log_path, error_info)
        except Exception as e:
            print(f"   ⚠️ Could not save failed conversion log: {e}")
    
    def _print_progress(self, current, total, start_time):
        """Print conversion progress."""
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        print(f"\n📊 Progress: {current}/{total} files")
        print(f"   ✅ Successful (in batch): {self.stats['successful']}")
        print(f"   ❌ Failed (in batch): {self.stats['failed']}")
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
        return self.stats.copy()


def create_document_converter(config):
    """Factory to create a DocumentConverter instance."""
    return DocumentConverter(config)