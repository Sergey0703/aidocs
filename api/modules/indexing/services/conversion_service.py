# api/modules/indexing/services/conversion_service.py
# Business logic for document conversion (Docling)

import asyncio
import logging
import time
import uuid
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
    """Service for managing document conversion operations"""
    
    def __init__(self):
        # Active tasks storage
        self._tasks: Dict[str, ConversionTaskState] = {}
        
        # Lock for task management
        self._lock = asyncio.Lock()
        
        logger.info("✅ ConversionService initialized")
    
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
        """Start conversion process in background"""
        
        task = await self.get_task(task_id)
        if not task:
            logger.error(f"❌ Task not found: {task_id}")
            return False
        
        # Update task status
        task.status = ConversionStatus.CONVERTING
        task.start_time = datetime.now()
        
        # Run conversion in background
        asyncio.create_task(
            self._run_conversion(
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
    
    async def _run_conversion(
        self,
        task: ConversionTaskState,
        input_dir: Optional[str],
        output_dir: Optional[str],
        incremental: bool,
        formats: Optional[List[str]],
        enable_ocr: Optional[bool],
        max_file_size_mb: Optional[int],
    ):
        """Run document conversion"""
        
        try:
            logger.info(f"🔄 Starting conversion for task: {task.task_id}")
            
            # Import Docling modules
            import sys
            from pathlib import Path as PathLib
            
            # Add rag_indexer to path
            indexer_path = PathLib(__file__).parent.parent.parent.parent.parent / "rag_indexer"
            if str(indexer_path) not in sys.path:
                sys.path.insert(0, str(indexer_path))
            
            # TODO: Import and use Docling processor
            # from docling_processor import (
            #     get_docling_config,
            #     create_document_scanner,
            #     create_document_converter
            # )
            
            # Placeholder implementation
            task.total_files = 5  # Placeholder
            
            for i in range(task.total_files):
                if task.cancelled:
                    break
                
                task.current_file = f"document_{i+1}.pdf"
                
                # Simulate conversion
                await asyncio.sleep(0.5)
                
                # Add result
                result = ConversionResult(
                    filename=task.current_file,
                    status=ConversionStatus.COMPLETED,
                    input_path=f"/input/{task.current_file}",
                    output_path=f"/output/document_{i+1}.md",
                    file_size=1024 * (i + 1),
                    conversion_time=0.5,
                )
                
                task.results.append(result)
                task.converted_files += 1
            
            # Complete task
            task.status = ConversionStatus.COMPLETED
            task.end_time = datetime.now()
            
            logger.info(f"✅ Conversion completed: {task.task_id}")
            
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


# Singleton instance
_conversion_service: Optional[ConversionService] = None


def get_conversion_service() -> ConversionService:
    """Get or create conversion service singleton"""
    global _conversion_service
    
    if _conversion_service is None:
        _conversion_service = ConversionService()
    
    return _conversion_service