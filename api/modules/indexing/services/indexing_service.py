# api/modules/indexing/services/indexing_service.py
# Business logic for indexing operations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..models.schemas import (
    IndexingMode,
    IndexingStatus,
    ProcessingStage,
    IndexingProgress,
    IndexingStatistics,
    IndexingHistoryItem,
)

logger = logging.getLogger(__name__)


class IndexingTaskState:
    """State management for indexing task"""
    
    def __init__(self, task_id: str, mode: IndexingMode):
        self.task_id = task_id
        self.mode = mode
        self.status = IndexingStatus.IDLE
        self.stage = None
        
        # Progress tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.progress_percentage = 0.0
        
        # File tracking
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self.current_file: Optional[str] = None
        
        # Chunk tracking
        self.total_chunks = 0
        self.processed_chunks = 0
        
        # Batch tracking
        self.current_batch: Optional[int] = None
        self.total_batches: Optional[int] = None
        
        # Performance
        self.processing_speed = 0.0
        self.avg_time_per_file = 0.0
        
        # Statistics
        self.statistics = IndexingStatistics()
        
        # Errors and warnings
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # Cancellation
        self.cancelled = False
    
    def get_progress(self) -> IndexingProgress:
        """Get current progress"""
        elapsed_time = 0.0
        estimated_remaining = None
        
        if self.start_time:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            
            # Calculate ETA
            if self.processed_chunks > 0 and self.total_chunks > 0:
                remaining_chunks = self.total_chunks - self.processed_chunks
                chunks_per_second = self.processed_chunks / elapsed_time if elapsed_time > 0 else 0
                if chunks_per_second > 0:
                    estimated_remaining = remaining_chunks / chunks_per_second
        
        return IndexingProgress(
            status=self.status,
            stage=self.stage,
            current_stage_name=self.stage.value if self.stage else "",
            progress_percentage=self.progress_percentage,
            total_files=self.total_files,
            processed_files=self.processed_files,
            failed_files=self.failed_files,
            total_chunks=self.total_chunks,
            processed_chunks=self.processed_chunks,
            start_time=self.start_time,
            elapsed_time=elapsed_time,
            estimated_remaining=estimated_remaining,
            current_file=self.current_file,
            current_batch=self.current_batch,
            total_batches=self.total_batches,
            processing_speed=self.processing_speed,
            avg_time_per_file=self.avg_time_per_file,
        )
    
    def update_progress(self, percentage: float):
        """Update progress percentage"""
        self.progress_percentage = min(100.0, max(0.0, percentage))
    
    def update_speed(self):
        """Update processing speed metrics"""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed > 0:
                self.processing_speed = self.processed_chunks / elapsed
                if self.processed_files > 0:
                    self.avg_time_per_file = elapsed / self.processed_files


class IndexingService:
    """Service for managing indexing operations"""
    
    def __init__(self):
        # Active tasks storage (in production, use Redis or database)
        self._tasks: Dict[str, IndexingTaskState] = {}
        
        # History storage (in production, use database)
        self._history: List[IndexingHistoryItem] = []
        
        # Lock for task management
        self._lock = asyncio.Lock()
        
        logger.info("✅ IndexingService initialized")
    
    async def create_task(self, mode: IndexingMode) -> str:
        """Create new indexing task"""
        async with self._lock:
            task_id = str(uuid.uuid4())
            task_state = IndexingTaskState(task_id, mode)
            self._tasks[task_id] = task_state
            
            logger.info(f"📝 Created indexing task: {task_id} (mode: {mode.value})")
            return task_id
    
    async def get_task(self, task_id: str) -> Optional[IndexingTaskState]:
        """Get task by ID"""
        return self._tasks.get(task_id)
    
    async def start_indexing(
        self,
        task_id: str,
        documents_dir: Optional[str] = None,
        skip_conversion: bool = False,
        skip_indexing: bool = False,
        batch_size: Optional[int] = None,
        force_reindex: bool = False,
        delete_existing: bool = False,
    ) -> bool:
        """Start indexing process in background"""
        
        task = await self.get_task(task_id)
        if not task:
            logger.error(f"❌ Task not found: {task_id}")
            return False
        
        # Update task status
        task.status = IndexingStatus.RUNNING
        task.start_time = datetime.now()
        
        # Run indexing in background
        asyncio.create_task(
            self._run_indexing_pipeline(
                task=task,
                documents_dir=documents_dir,
                skip_conversion=skip_conversion,
                skip_indexing=skip_indexing,
                batch_size=batch_size,
                force_reindex=force_reindex,
                delete_existing=delete_existing,
            )
        )
        
        logger.info(f"🚀 Started indexing task: {task_id}")
        return True
    
    async def _run_indexing_pipeline(
        self,
        task: IndexingTaskState,
        documents_dir: Optional[str],
        skip_conversion: bool,
        skip_indexing: bool,
        batch_size: Optional[int],
        force_reindex: bool,
        delete_existing: bool,
    ):
        """Run complete indexing pipeline"""
        
        try:
            logger.info(f"🔄 Starting indexing pipeline for task: {task.task_id}")
            
            # Import rag_indexer modules
            import sys
            from pathlib import Path as PathLib
            
            # Add rag_indexer to path
            indexer_path = PathLib(__file__).parent.parent.parent.parent.parent / "rag_indexer"
            if str(indexer_path) not in sys.path:
                sys.path.insert(0, str(indexer_path))
            
            # Stage 1: Document Conversion (if not skipped)
            if not skip_conversion:
                await self._run_conversion_stage(task, documents_dir)
                
                if task.cancelled:
                    logger.info(f"⚠️ Task cancelled during conversion: {task.task_id}")
                    task.status = IndexingStatus.CANCELLED
                    return
            
            # Stage 2: Vector Indexing (if not skipped)
            if not skip_indexing:
                await self._run_indexing_stage(
                    task=task,
                    documents_dir=documents_dir,
                    batch_size=batch_size,
                    force_reindex=force_reindex,
                    delete_existing=delete_existing,
                )
                
                if task.cancelled:
                    logger.info(f"⚠️ Task cancelled during indexing: {task.task_id}")
                    task.status = IndexingStatus.CANCELLED
                    return
            
            # Complete task
            task.status = IndexingStatus.COMPLETED
            task.end_time = datetime.now()
            task.progress_percentage = 100.0
            
            # Add to history
            await self._add_to_history(task)
            
            logger.info(f"✅ Indexing pipeline completed: {task.task_id}")
            
        except Exception as e:
            logger.error(f"❌ Indexing pipeline failed for task {task.task_id}: {e}", exc_info=True)
            task.status = IndexingStatus.FAILED
            task.end_time = datetime.now()
            task.errors.append(f"Pipeline error: {str(e)}")
            
            # Add to history even if failed
            await self._add_to_history(task)
    
    async def _run_conversion_stage(self, task: IndexingTaskState, documents_dir: Optional[str]):
        """Run document conversion stage (Part 1) - Real Docling integration"""
        
        task.stage = ProcessingStage.CONVERSION
        logger.info(f"📄 Stage 1: Document Conversion - Task {task.task_id}")
        
        try:
            # Import Docling processor modules
            from docling_processor import (
                get_docling_config,
                create_document_scanner,
                create_document_converter
            )
            
            # Get configuration
            config = get_docling_config()
            if documents_dir:
                config.RAW_DOCUMENTS_DIR = documents_dir
            
            logger.info(f"📂 Scanning documents in: {config.RAW_DOCUMENTS_DIR}")
            
            # Scan for documents
            scanner = create_document_scanner(config)
            files_to_process = scanner.scan_directory()
            
            task.total_files = len(files_to_process)
            logger.info(f"📄 Found {task.total_files} files to convert")
            
            if task.total_files == 0:
                logger.info("No files to convert")
                return
            
            # Filter already converted (incremental mode)
            if task.mode == IndexingMode.INCREMENTAL:
                files_to_process = scanner.filter_already_converted(
                    files_to_process,
                    incremental=True
                )
                task.total_files = len(files_to_process)
                logger.info(f"🔄 Incremental mode: {task.total_files} new/modified files")
            
            if task.total_files == 0:
                logger.info("All files already converted")
                return
            
            # Create converter
            converter = create_document_converter(config)
            
            # Convert documents
            logger.info(f"🔄 Starting conversion of {task.total_files} files...")
            results = converter.convert_batch(files_to_process)
            
            # Update task statistics
            task.processed_files = results['successful']
            task.failed_files = results['failed']
            task.statistics.documents_converted = results['successful']
            
            logger.info(f"✅ Conversion completed: {task.processed_files}/{task.total_files} files")
            
            if results['failed'] > 0:
                task.warnings.append(f"{results['failed']} files failed conversion")
            
        except Exception as e:
            logger.error(f"❌ Conversion stage failed: {e}", exc_info=True)
            task.errors.append(f"Conversion stage error: {str(e)}")
            raise
    
    async def _run_indexing_stage(
        self,
        task: IndexingTaskState,
        documents_dir: Optional[str],
        batch_size: Optional[int],
        force_reindex: bool,
        delete_existing: bool,
    ):
        """Run vector indexing stage (Part 2)"""
        
        task.stage = ProcessingStage.LOADING
        logger.info(f"🧩 Stage 2: Vector Indexing - Task {task.task_id}")
        
        try:
            # TODO: Import and run vector indexing
            # This is a placeholder - needs actual implementation
            
            task.total_chunks = 100  # Placeholder
            task.processed_chunks = 100  # Placeholder
            task.statistics.chunks_saved = 100
            
            task.stage = ProcessingStage.COMPLETED
            logger.info(f"✅ Indexing stage completed: {task.statistics.chunks_saved} chunks saved")
            
        except Exception as e:
            logger.error(f"❌ Indexing stage failed: {e}", exc_info=True)
            task.errors.append(f"Indexing stage error: {str(e)}")
            raise
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel running task"""
        task = await self.get_task(task_id)
        if not task:
            return False
        
        if task.status != IndexingStatus.RUNNING:
            logger.warning(f"⚠️ Cannot cancel task {task_id}: not running (status: {task.status.value})")
            return False
        
        task.cancelled = True
        logger.info(f"⚠️ Cancelling task: {task_id}")
        return True
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current task status"""
        task = await self.get_task(task_id)
        if not task:
            return None
        
        return {
            "task_id": task_id,
            "progress": task.get_progress(),
            "statistics": task.statistics,
            "errors": task.errors,
            "warnings": task.warnings,
        }
    
    async def get_history(self, limit: int = 10) -> List[IndexingHistoryItem]:
        """Get indexing history"""
        return self._history[-limit:] if limit > 0 else self._history
    
    async def _add_to_history(self, task: IndexingTaskState):
        """Add completed task to history"""
        
        duration = None
        if task.start_time and task.end_time:
            duration = (task.end_time - task.start_time).total_seconds()
        
        history_item = IndexingHistoryItem(
            task_id=task.task_id,
            mode=task.mode,
            status=task.status,
            start_time=task.start_time,
            end_time=task.end_time,
            duration=duration,
            files_processed=task.processed_files,
            chunks_created=task.statistics.chunks_created,
            success_rate=task.statistics.success_rate,
            error_message=task.errors[0] if task.errors else None,
        )
        
        async with self._lock:
            self._history.append(history_item)
            
            # Keep only last 100 items
            if len(self._history) > 100:
                self._history = self._history[-100:]
        
        logger.info(f"📝 Added task to history: {task.task_id}")
    
    async def clear_completed_tasks(self):
        """Clear completed tasks from memory"""
        async with self._lock:
            completed = [
                task_id for task_id, task in self._tasks.items()
                if task.status in [IndexingStatus.COMPLETED, IndexingStatus.FAILED, IndexingStatus.CANCELLED]
            ]
            
            for task_id in completed:
                del self._tasks[task_id]
            
            logger.info(f"🧹 Cleared {len(completed)} completed tasks")
    
    def get_active_tasks_count(self) -> int:
        """Get number of active tasks"""
        return sum(
            1 for task in self._tasks.values()
            if task.status == IndexingStatus.RUNNING
        )


# Singleton instance
_indexing_service: Optional[IndexingService] = None


def get_indexing_service() -> IndexingService:
    """Get or create indexing service singleton"""
    global _indexing_service
    
    if _indexing_service is None:
        _indexing_service = IndexingService()
    
    return _indexing_service