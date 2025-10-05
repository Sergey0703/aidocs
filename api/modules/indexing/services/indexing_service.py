#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# api/modules/indexing/services/indexing_service.py
# Real implementation with backend integration

import asyncio
import logging
import time
import uuid
import sys
import os
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
        
        # Progress callback for real-time updates
        self.progress_callback = None
    
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
        
        # Trigger callback if set
        if self.progress_callback:
            try:
                self.progress_callback(self.get_progress())
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def update_speed(self):
        """Update processing speed metrics"""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed > 0:
                self.processing_speed = self.processed_chunks / elapsed
                if self.processed_files > 0:
                    self.avg_time_per_file = elapsed / self.processed_files


class IndexingService:
    """Service for managing indexing operations with real backend integration"""
    
    def __init__(self):
        # Active tasks storage (for small scale, in-memory is fine)
        self._tasks: Dict[str, IndexingTaskState] = {}
        
        # History storage (keep last 100 runs)
        self._history: List[IndexingHistoryItem] = []
        
        # Lock for task management
        self._lock = asyncio.Lock()
        
        logger.info("✅ IndexingService initialized with backend integration")
    
    def _setup_backend_path(self):
        """Add rag_indexer to Python path"""
        try:
            # Get path to rag_indexer directory
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent.parent
            backend_path = project_root / "rag_indexer"
            
            if backend_path.exists():
                sys.path.insert(0, str(backend_path))
                logger.info(f"Added backend path: {backend_path}")
            else:
                logger.warning(f"Backend path not found: {backend_path}")
        except Exception as e:
            logger.error(f"Failed to setup backend path: {e}")
    
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
        """Start indexing process - calls real backend code"""
        
        task = await self.get_task(task_id)
        if not task:
            logger.error(f"❌ Task not found: {task_id}")
            return False
        
        # Update task status
        task.status = IndexingStatus.RUNNING
        task.start_time = datetime.now()
        
        # Run indexing in background
        asyncio.create_task(
            self._run_real_indexing_pipeline(
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
    
    async def _run_real_indexing_pipeline(
        self,
        task: IndexingTaskState,
        documents_dir: Optional[str],
        skip_conversion: bool,
        skip_indexing: bool,
        batch_size: Optional[int],
        force_reindex: bool,
        delete_existing: bool,
    ):
        """Run REAL indexing pipeline using backend code"""
        
        try:
            logger.info(f"🔄 Starting REAL indexing pipeline for task: {task.task_id}")
            
            # Add backend path to sys.path just-in-time
            self._setup_backend_path()
            
            # Import real backend modules
            from chunking_vectors.config import get_config
            from chunking_vectors.database_manager import create_database_manager
            from chunking_vectors.loading_helpers import load_markdown_documents
            from chunking_vectors.chunk_helpers import create_and_filter_chunks_enhanced
            from chunking_vectors.embedding_processor import create_embedding_processor
            from chunking_vectors.batch_processor import create_batch_processor, create_progress_tracker
            from chunking_vectors.markdown_loader import create_markdown_loader
            
            from llama_index.core import StorageContext
            from llama_index.vector_stores.supabase import SupabaseVectorStore
            from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
            from llama_index.core.node_parser import SentenceSplitter
            
            # Load configuration
            task.stage = ProcessingStage.CONVERSION
            task.update_progress(5)
            
            config = get_config()
            
            # Override config if provided
            if documents_dir:
                config.DOCUMENTS_DIR = documents_dir
            if batch_size:
                config.PROCESSING_BATCH_SIZE = batch_size
            
            logger.info(f"📁 Documents directory: {config.DOCUMENTS_DIR}")
            
            # Initialize components
            task.update_progress(10)
            
            # Vector store
            vector_store = SupabaseVectorStore(
                postgres_connection_string=config.CONNECTION_STRING,
                collection_name=config.TABLE_NAME,
                dimension=config.EMBED_DIM,
            )
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Gemini embedding model
            embed_model = GoogleGenAIEmbedding(
                model_name=config.EMBED_MODEL,
                api_key=config.GEMINI_API_KEY,
            )
            
            # Node parser
            node_parser = SentenceSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                paragraph_separator="\n\n",
                include_metadata=True,
            )
            
            # Database manager
            db_manager = create_database_manager(
                config.CONNECTION_STRING,
                config.TABLE_NAME
            )
            
            logger.info("✅ Components initialized")
            
            # ============================================================
            # STAGE 1: Load markdown documents
            # ============================================================
            
            if not skip_conversion:
                task.stage = ProcessingStage.LOADING
                task.update_progress(15)
                
                logger.info("📄 Loading markdown documents...")
                
                progress_tracker = create_progress_tracker()
                progress_tracker.start()
                
                documents, processing_summary = load_markdown_documents(
                    config, 
                    progress_tracker
                )
                
                task.total_files = len(documents)
                task.statistics.documents_loaded = len(documents)
                
                logger.info(f"✅ Loaded {len(documents)} documents")
                
                if not documents:
                    raise Exception("No documents found to process")
            else:
                logger.info("⏩ Skipping document loading (skip_conversion=True)")
                task.update_progress(30)
            
            # ============================================================
            # STAGE 2: Handle deletion if requested
            # ============================================================
            
            if delete_existing and not skip_conversion:
                task.update_progress(20)
                
                logger.info("🗑️ Handling existing records...")
                
                files_to_process = set()
                for doc in documents:
                    file_path = doc.metadata.get('file_path', '')
                    if file_path:
                        files_to_process.add(file_path)
                
                # Delete existing records
                deleted_count = db_manager.delete_existing_records(files_to_process)
                logger.info(f"🗑️ Deleted {deleted_count} existing records")
            
            # ============================================================
            # STAGE 3: Create and filter chunks
            # ============================================================
            
            if not skip_indexing and not skip_conversion:
                task.stage = ProcessingStage.CHUNKING
                task.update_progress(30)
                
                logger.info("🧩 Creating and filtering chunks...")
                
                valid_nodes, invalid_nodes, node_stats = create_and_filter_chunks_enhanced(
                    documents,
                    config,
                    node_parser,
                    progress_tracker
                )
                
                task.total_chunks = len(valid_nodes)
                task.statistics.chunks_created = node_stats['total_nodes_created']
                task.statistics.chunks_valid = len(valid_nodes)
                task.statistics.chunks_invalid = len(invalid_nodes)
                
                logger.info(f"✅ Created {len(valid_nodes)} valid chunks")
                
                if not valid_nodes:
                    raise Exception("No valid chunks generated")
                
                task.update_progress(40)
            else:
                logger.info("⏩ Skipping chunking")
                task.update_progress(60)
                valid_nodes = []
            
            # ============================================================
            # STAGE 4: Generate embeddings and save to database
            # ============================================================
            
            if not skip_indexing and valid_nodes:
                task.stage = ProcessingStage.EMBEDDING
                task.update_progress(50)
                
                logger.info("🚀 Starting embedding generation and database save...")
                
                # Create embedding processor
                embedding_processor = create_embedding_processor(
                    embed_model,
                    vector_store,
                    config
                )
                
                # Create batch processor
                batch_processor = create_batch_processor(
                    embedding_processor,
                    config.PROCESSING_BATCH_SIZE,
                    batch_restart_interval=0,  # No restarts needed for Gemini
                    config=config
                )
                
                # Set up progress callback
                def progress_callback(current_chunks, total_chunks):
                    task.processed_chunks = current_chunks
                    progress_pct = 50 + (current_chunks / total_chunks * 40)  # 50-90%
                    task.update_progress(progress_pct)
                    task.update_speed()
                
                # Process all batches
                batch_results = batch_processor.process_all_batches(
                    valid_nodes,
                    config.EMBEDDING_BATCH_SIZE,
                    config.DB_BATCH_SIZE
                )
                
                # Update statistics
                task.statistics.chunks_saved = batch_results['total_saved']
                task.statistics.success_rate = batch_results['success_rate']
                task.statistics.gemini_api_calls = batch_results.get('gemini_api_calls', 0)
                task.statistics.gemini_tokens_used = batch_results.get('total_saved', 0) * 100  # Rough estimate
                
                task.processed_chunks = batch_results['total_saved']
                task.processed_files = task.total_files
                
                logger.info(f"✅ Saved {batch_results['total_saved']} chunks to database")
                
                if batch_results.get('failed_batches', 0) > 0:
                    task.warnings.append(f"{batch_results['failed_batches']} batches had errors")
                
                task.update_progress(95)
            else:
                logger.info("⏩ Skipping embedding generation")
                task.update_progress(90)
            
            # ============================================================
            # COMPLETE
            # ============================================================
            
            task.stage = ProcessingStage.COMPLETED
            task.status = IndexingStatus.COMPLETED
            task.end_time = datetime.now()
            task.update_progress(100)
            
            # Calculate final statistics
            if task.start_time and task.end_time:
                duration = (task.end_time - task.start_time).total_seconds()
                task.statistics.total_time = duration
                
                if task.processed_chunks > 0:
                    task.statistics.avg_chunk_length = task.processed_chunks / duration if duration > 0 else 0
            
            # Add to history
            await self._add_to_history(task)
            
            logger.info(f"✅ REAL indexing pipeline completed: {task.task_id}")
            
        except asyncio.CancelledError:
            logger.info(f"⚠️ Task cancelled: {task.task_id}")
            task.status = IndexingStatus.CANCELLED
            task.end_time = datetime.now()
            await self._add_to_history(task)
            
        except Exception as e:
            logger.error(f"❌ REAL indexing pipeline failed for task {task.task_id}: {e}", exc_info=True)
            task.status = IndexingStatus.FAILED
            task.end_time = datetime.now()
            task.errors.append(f"Pipeline error: {str(e)}")
            
            # Add to history even if failed
            await self._add_to_history(task)
    
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
            "timestamp": datetime.now()
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
    
    async def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks (active and completed)"""
        tasks = []
        
        for task_id, task in self._tasks.items():
            tasks.append({
                "task_id": task_id,
                "mode": task.mode.value,
                "status": task.status.value,
                "progress": task.progress_percentage,
                "start_time": task.start_time,
                "end_time": task.end_time,
            })
        
        return tasks


# Singleton instance
_indexing_service: Optional[IndexingService] = None


def get_indexing_service() -> IndexingService:
    """Get or create indexing service singleton"""
    global _indexing_service
    
    if _indexing_service is None:
        _indexing_service = IndexingService()
    
    return _indexing_service