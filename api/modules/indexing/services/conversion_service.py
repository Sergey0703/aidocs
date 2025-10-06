#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# api/modules/indexing/services/conversion_service.py
# Real Docling integration for document conversion

import asyncio
import logging
import time
import uuid
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..models.schemas import (
    ConversionStatus,
    ConversionProgress,
    ConversionResult,
)

logger = logging.getLogger(__name__)


class ConversionTaskState:
    """State management for conversion task"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = ConversionStatus.PENDING
        
        # Progress tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # File tracking
        self.total_files = 0
        self.converted_files = 0
        self.failed_files = 0
        self.skipped_files = 0
        self.current_file: Optional[str] = None
        
        # Results
        self.results: List[ConversionResult] = []
        
        # Errors
        self.errors: List[str] = []
        
        # Cancellation
        self.cancelled = False
    
    def get_progress(self) -> ConversionProgress:
        """Get current progress"""
        elapsed_time = 0.0
        estimated_remaining = None
        progress_percentage = 0.0
        
        if self.start_time:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            
            # Calculate progress
            if self.total_files > 0:
                completed = self.converted_files + self.failed_files + self.skipped_files
                progress_percentage = (completed / self.total_files) * 100
                
                # Calculate ETA
                if completed > 0 and elapsed_time > 0:
                    remaining_files = self.total_files - completed
                    files_per_second = completed / elapsed_time
                    if files_per_second > 0:
                        estimated_remaining = remaining_files / files_per_second
        
        return ConversionProgress(
            status=self.status,
            total_files=self.total_files,
            converted_files=self.converted_files,
            failed_files=self.failed_files,
            skipped_files=self.skipped_files,
            progress_percentage=progress_percentage,
            current_file=self.current_file,
            elapsed_time=elapsed_time,
            estimated_remaining=estimated_remaining,
        )


class ConversionService:
    """Service for managing document conversion operations with real Docling"""
    
    def __init__(self):
        # Active tasks storage
        self._tasks: Dict[str, ConversionTaskState] = {}
        
        # Lock for task management
        self._lock = asyncio.Lock()
        
        logger.info("✅ ConversionService initialized with Docling integration")
    
    def _setup_backend_path(self):
        """Add rag_indexer to Python path"""
        try:
            # Get path to rag_indexer directory
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent.parent
            backend_path = project_root / "rag_indexer"
            
            if backend_path.exists() and str(backend_path) not in sys.path:
                sys.path.insert(0, str(backend_path))
                logger.info(f"Added backend path: {backend_path}")
            elif not backend_path.exists():
                logger.warning(f"Backend path not found: {backend_path}")
        except Exception as e:
            logger.error(f"Failed to setup backend path: {e}")
    
    async def create_task(self) -> str:
        """Create new conversion task"""
        async with self._lock:
            task_id = "conv_" + str(uuid.uuid4())
            task_state = ConversionTaskState(task_id)
            self._tasks[task_id] = task_state
            
            logger.info(f"📝 Created conversion task: {task_id}")
            return task_id
    
    async def get_task(self, task_id: str) -> Optional[ConversionTaskState]:
        """Get task by ID"""
        return self._tasks.get(task_id)
    
    async def start_conversion(
        self,
        task_id: str,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        incremental: bool = True,
        formats: Optional[List[str]] = None,
        enable_ocr: Optional[bool] = None,
        max_file_size_mb: Optional[int] = None,
    ) -> bool:
        """Start conversion process using real Docling"""
        
        task = await self.get_task(task_id)
        if not task:
            logger.error(f"❌ Task not found: {task_id}")
            return False
        
        # Update task status
        task.status = ConversionStatus.CONVERTING
        task.start_time = datetime.now()
        
        # Run conversion in background
        asyncio.create_task(
            self._run_real_conversion(
                task=task,
                input_dir=input_dir,
                output_dir=output_dir,
                incremental=incremental,
                formats=formats,
                enable_ocr=enable_ocr,
                max_file_size_mb=max_file_size_mb,
            )
        )
        
        logger.info(f"🚀 Started conversion task: {task_id}")
        return True
    
    async def _run_real_conversion(
        self,
        task: ConversionTaskState,
        input_dir: Optional[str],
        output_dir: Optional[str],
        incremental: bool,
        formats: Optional[List[str]],
        enable_ocr: Optional[bool],
        max_file_size_mb: Optional[int],
    ):
        """Run REAL document conversion using Docling"""
        
        try:
            logger.info(f"🔄 Starting REAL conversion for task: {task.task_id}")
            
            # Add backend path to sys.path just-in-time
            self._setup_backend_path()
            
            # Import real Docling modules
            from docling_processor import (
                get_docling_config,
                create_document_scanner,
                create_document_converter
            )
            
            # Load Docling configuration
            config = get_docling_config()
            
            # --- FIX: Resolve relative paths to absolute paths ---
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent.parent
            
            # Default raw documents directory
            default_raw_dir = project_root / "rag_indexer" / "data" / "raw"
            default_raw_dir.mkdir(parents=True, exist_ok=True)
            
            # Default markdown output directory
            default_md_dir = project_root / "rag_indexer" / "data" / "markdown"
            default_md_dir.mkdir(parents=True, exist_ok=True)

            # --- START: ИСПРАВЛЕНИЕ ---
            # Явно определяем путь для метаданных
            default_metadata_dir = default_md_dir / "_metadata"
            # --- END: ИСПРАВЛЕНИЕ ---
            
            # Override config if provided
            config.RAW_DOCUMENTS_DIR = input_dir if input_dir else str(default_raw_dir)
            config.MARKDOWN_OUTPUT_DIR = output_dir if output_dir else str(default_md_dir)

            # --- START: ИСПРАВЛЕНИЕ ---
            # Явно устанавливаем путь к метаданным в конфигурации
            config.METADATA_DIR = str(default_metadata_dir)
            # --- END: ИСПРАВЛЕНИЕ ---
            
            if enable_ocr is not None:
                config.ENABLE_OCR = enable_ocr
            if max_file_size_mb is not None:
                config.MAX_FILE_SIZE_MB = max_file_size_mb
            if formats:
                config.SUPPORTED_FORMATS = formats
            
            # Логирование для отладки
            logger.info(f"📁 Input directory: {config.RAW_DOCUMENTS_DIR}")
            logger.info(f"📁 Output directory: {config.MARKDOWN_OUTPUT_DIR}")
            logger.info(f"📋 Metadata directory: {config.METADATA_DIR}")
            logger.info(f"📊 OCR enabled: {config.ENABLE_OCR}")
            
            # ========================================================
            # STAGE 1: Scan for documents
            # ========================================================
            
            logger.info("📂 Scanning for documents...")
            
            scanner = create_document_scanner(config)
            files_to_process = scanner.scan_directory()
            
            if not files_to_process:
                logger.info("⚠️ No files found to convert")
                task.status = ConversionStatus.COMPLETED
                task.end_time = datetime.now()
                return
            
            logger.info(f"✅ Found {len(files_to_process)} files")
            
            # ========================================================
            # STAGE 2: Filter already converted (if incremental)
            # ========================================================
            
            if incremental:
                logger.info("🔍 Filtering already converted files...")
                
                files_to_process = scanner.filter_already_converted(
                    files_to_process,
                    incremental=True
                )
                
                if not files_to_process:
                    logger.info("✅ All files already converted")
                    task.status = ConversionStatus.COMPLETED
                    task.end_time = datetime.now()
                    return
                
                logger.info(f"📄 {len(files_to_process)} files need conversion")
            
            task.total_files = len(files_to_process)
            
            # ========================================================
            # STAGE 3: Convert documents using Docling
            # ========================================================
            
            logger.info(f"🔄 Converting {len(files_to_process)} documents...")
            
            converter = create_document_converter(config)
            
            # Convert each file and track progress
            for i, file_path in enumerate(files_to_process, 1):
                if task.cancelled:
                    logger.info(f"⚠️ Conversion cancelled by user")
                    task.status = ConversionStatus.FAILED
                    task.errors.append("Cancelled by user")
                    break
                
                task.current_file = str(file_path.name)
                
                logger.info(f"[{i}/{len(files_to_process)}] Converting: {file_path.name}")
                
                # Convert single file
                conversion_start = time.time()
                success, output_path, error = converter.convert_file(file_path)
                conversion_time = time.time() - conversion_start
                
                # Create result
                result = ConversionResult(
                    filename=file_path.name,
                    status=ConversionStatus.COMPLETED if success else ConversionStatus.FAILED,
                    input_path=str(file_path),
                    output_path=str(output_path) if output_path else None,
                    file_size=file_path.stat().st_size,
                    conversion_time=conversion_time,
                    error_message=error,
                )
                
                task.results.append(result)
                
                if success:
                    task.converted_files += 1
                    logger.info(f"  ✅ Success ({conversion_time:.2f}s)")
                else:
                    task.failed_files += 1
                    logger.error(f"  ❌ Failed: {error}")
                    task.errors.append(f"{file_path.name}: {error}")
            
            # ========================================================
            # COMPLETE
            # ========================================================
            
            task.status = ConversionStatus.COMPLETED
            task.end_time = datetime.now()
            task.current_file = None
            
            # Summary
            if task.start_time:
                duration = (task.end_time - task.start_time).total_seconds()
            else:
                duration = 0
            
            logger.info(f"✅ Conversion completed: {task.task_id}")
            logger.info(f"   Total: {task.total_files}")
            logger.info(f"   ✅ Converted: {task.converted_files}")
            logger.info(f"   ❌ Failed: {task.failed_files}")
            logger.info(f"   ⏱️ Time: {duration:.1f}s")
            
            # Get conversion statistics from converter
            conv_stats = converter.get_conversion_stats()
            logger.info(f"   📊 Success rate: {conv_stats.get('successful', 0)}/{conv_stats.get('total_files', 0)}")
            
        except asyncio.CancelledError:
            logger.info(f"⚠️ Conversion task cancelled: {task.task_id}")
            task.status = ConversionStatus.FAILED
            task.end_time = datetime.now()
            task.errors.append("Task cancelled")
            
        except Exception as e:
            logger.error(f"❌ Conversion failed for task {task.task_id}: {e}", exc_info=True)
            task.status = ConversionStatus.FAILED
            task.end_time = datetime.now()
            task.errors.append(f"Conversion error: {str(e)}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel running conversion task"""
        task = await self.get_task(task_id)
        if not task:
            return False
        
        if task.status != ConversionStatus.CONVERTING:
            logger.warning(f"⚠️ Cannot cancel task {task_id}: not converting")
            return False
        
        task.cancelled = True
        logger.info(f"⚠️ Cancelling conversion task: {task_id}")
        return True
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current task status"""
        task = await self.get_task(task_id)
        if not task:
            return None
        
        return {
            "task_id": task_id,
            "progress": task.get_progress(),
            "results": task.results,
            "errors": task.errors,
            "timestamp": datetime.now()
        }
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete conversion task"""
        async with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                logger.info(f"🗑️ Deleted conversion task: {task_id}")
                return True
            return False
    
    def get_active_tasks_count(self) -> int:
        """Get number of active conversion tasks"""
        return sum(
            1 for task in self._tasks.values()
            if task.status == ConversionStatus.CONVERTING
        )
    
    async def get_supported_formats(self) -> Dict[str, Any]:
        """Get supported document formats from Docling config"""
        try:
            self._setup_backend_path()
            from docling_processor import get_docling_config
            
            config = get_docling_config()
            
            return {
                "formats": config.SUPPORTED_FORMATS,
                "ocr_enabled": config.ENABLE_OCR,
                "max_file_size_mb": config.MAX_FILE_SIZE_MB,
                "extract_tables": config.EXTRACT_TABLES,
                "ocr_language": config.OCR_LANGUAGE if config.ENABLE_OCR else None,
            }
            
        except Exception as e:
            logger.error(f"Failed to get supported formats: {e}")
            return {
                "formats": ["pdf", "docx", "pptx", "html", "txt"],
                "ocr_enabled": True,
                "max_file_size_mb": 100,
            }
    
    async def validate_documents(
        self,
        input_dir: Optional[str] = None,
        check_formats: bool = True,
        check_size: bool = True,
        max_size_mb: Optional[int] = None
    ) -> Dict[str, Any]:
        """Validate documents before conversion"""
        try:
            self._setup_backend_path()
            from docling_processor import get_docling_config, create_document_scanner
            
            config = get_docling_config()
            
            if input_dir:
                config.RAW_DOCUMENTS_DIR = input_dir
            if max_size_mb:
                config.MAX_FILE_SIZE_MB = max_size_mb
            
            # Scan directory
            scanner = create_document_scanner(config)
            scanner.scan_directory()
            
            # Get scan stats
            scan_stats = scanner.get_scan_stats()
            
            validation_result = {
                "valid": True,
                "total_files": scan_stats.get('total_files', 0),
                "supported_files": scan_stats.get('supported_files', 0),
                "unsupported_files": scan_stats.get('unsupported_files', 0),
                "oversized_files": scan_stats.get('oversized_files', 0),
                "by_format": scan_stats.get('by_format', {}),
                "warnings": [],
                "errors": []
            }
            
            # Add warnings
            if validation_result['unsupported_files'] > 0:
                validation_result['warnings'].append(
                    f"{validation_result['unsupported_files']} files have unsupported formats"
                )
            
            if validation_result['oversized_files'] > 0:
                validation_result['warnings'].append(
                    f"{validation_result['oversized_files']} files exceed size limit"
                )
            
            if validation_result['supported_files'] == 0:
                validation_result['valid'] = False
                validation_result['errors'].append("No supported files found")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "valid": False,
                "errors": [str(e)]
            }
    
    async def get_conversion_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get history of conversion tasks"""
        try:
            # Get completed tasks
            history = []
            
            # Sort tasks by start time to get the most recent ones
            sorted_tasks = sorted(
                self._tasks.items(),
                key=lambda item: item[1].start_time if item[1].start_time else datetime.min,
                reverse=True
            )

            for task_id, task in sorted_tasks[:limit]:
                if task.status in [ConversionStatus.COMPLETED, ConversionStatus.FAILED]:
                    history.append({
                        "task_id": task_id,
                        "status": task.status.value,
                        "total_files": task.total_files,
                        "converted_files": task.converted_files,
                        "failed_files": task.failed_files,
                        "start_time": task.start_time,
                        "end_time": task.end_time,
                        "duration": (task.end_time - task.start_time).total_seconds() if task.start_time and task.end_time else None,
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get conversion history: {e}")
            return []
    
    async def retry_failed_conversions(
        self,
        original_task_id: str
    ) -> Optional[str]:
        """Retry failed conversions from a previous task"""
        try:
            # Get original task
            original_task = await self.get_task(original_task_id)
            if not original_task:
                logger.error(f"Original task not found: {original_task_id}")
                return None
            
            # Get failed files
            failed_results = [
                r for r in original_task.results
                if r.status == ConversionStatus.FAILED
            ]
            
            if not failed_results:
                logger.info("No failed files to retry")
                return None
            
            logger.info(f"Found {len(failed_results)} failed files to retry")
            
            # Create new task
            new_task_id = await self.create_task()
            
            # Start conversion with same settings but only failed files
            # For now, just return the task_id - actual retry logic would need
            # to filter files in the scanner
            
            logger.info(f"Created retry task: {new_task_id}")
            return new_task_id
            
        except Exception as e:
            logger.error(f"Failed to create retry task: {e}")
            return None


# Singleton instance
_conversion_service: Optional[ConversionService] = None


def get_conversion_service() -> ConversionService:
    """Get or create conversion service singleton"""
    global _conversion_service
    
    if _conversion_service is None:
        _conversion_service = ConversionService()
    
    return _conversion_service