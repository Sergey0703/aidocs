#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# api/modules/indexing/services/indexing_service.py
# Fixed: skip_conversion now only skips Docling, NOT the entire indexing pipeline

import asyncio
import logging
import time
import uuid
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..models.schemas import (
    IndexingMode,
    IndexingStatus,
    IndexingProgress,
    IndexingHistoryItem,
)

logger = logging.getLogger(__name__)


class IndexingTaskState:
    """State management for indexing task"""
    
    def __init__(self, task_id: str, mode: IndexingMode):
        self.task_id = task_id
        self.mode = mode
        self.status = IndexingStatus.IDLE
        
        # Progress tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # File/chunk tracking
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self.total_chunks = 0
        self.processed_chunks = 0
        self.current_file: Optional[str] = None
        
        # Stage tracking
        self.current_stage: Optional[str] = None
        self.current_stage_name: Optional[str] = None
        
        # Statistics
        self.statistics: Dict[str, Any] = {}
        
        # Errors
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # Cancellation
        self.cancelled = False
    
    def get_progress(self) -> IndexingProgress:
        """Get current progress"""
        elapsed_time = 0.0
        estimated_remaining = None
        progress_percentage = 0.0
        
        if self.start_time:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            
            # Calculate progress based on chunks
            if self.total_chunks > 0:
                progress_percentage = (self.processed_chunks / self.total_chunks) * 100
                
                # Calculate ETA
                if self.processed_chunks > 0 and elapsed_time > 0:
                    remaining_chunks = self.total_chunks - self.processed_chunks
                    chunks_per_second = self.processed_chunks / elapsed_time
                    if chunks_per_second > 0:
                        estimated_remaining = remaining_chunks / chunks_per_second
        
        return IndexingProgress(
            status=self.status,
            total_files=self.total_files,
            processed_files=self.processed_files,
            failed_files=self.failed_files,
            total_chunks=self.total_chunks,
            processed_chunks=self.processed_chunks,
            progress_percentage=progress_percentage,
            current_file=self.current_file,
            current_stage=self.current_stage,
            current_stage_name=self.current_stage_name,
            elapsed_time=elapsed_time,
            estimated_remaining=estimated_remaining,
        )


class IndexingService:
    """Service for managing document indexing operations with real backend integration"""
    
    def __init__(self):
        # Active tasks storage
        self._tasks: Dict[str, IndexingTaskState] = {}
        
        # History storage (keep last 50 tasks)
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
        """Start indexing process using real backend"""
        
        task = await self.get_task(task_id)
        if not task:
            logger.error(f"❌ Task not found: {task_id}")
            return False
        
        # Update task status
        task.status = IndexingStatus.RUNNING
        task.start_time = datetime.now()
        
        # Run indexing in background
        asyncio.create_task(
            self._run_real_indexing(
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
    
    async def _run_real_indexing(
        self,
        task: IndexingTaskState,
        documents_dir: Optional[str],
        skip_conversion: bool,
        skip_indexing: bool,
        batch_size: Optional[int],
        force_reindex: bool,
        delete_existing: bool,
    ):
        """Run REAL document indexing using backend modules"""
        
        try:
            logger.info(f"🔄 Starting REAL indexing pipeline for task: {task.task_id}")
            
            # Add backend path to sys.path just-in-time
            self._setup_backend_path()
            
            # Import real backend modules
            from chunking_vectors.config import get_config
            from chunking_vectors.document_loader import DocumentLoader
            from chunking_vectors.chunker import DocumentChunker
            from chunking_vectors.embedding_generator import EmbeddingGenerator
            from chunking_vectors.database_manager import create_database_manager
            
            # Load configuration
            config = get_config()
            
            # Override documents directory if provided
            if documents_dir:
                config.DOCUMENTS_DIR = documents_dir
            
            logger.info(f"📁 Documents directory: {config.DOCUMENTS_DIR}")
            
            # Test database connection
            try:
                import psycopg2
                conn = psycopg2.connect(config.CONNECTION_STRING)
                conn.close()
                logger.info("Database connection: SUCCESS")
            except Exception as e:
                logger.error(f"Database connection: FAILED - {e}")
                task.errors.append(f"Database connection failed: {str(e)}")
                task.status = IndexingStatus.FAILED
                task.end_time = datetime.now()
                return
            
            # Initialize components
            doc_loader = DocumentLoader(config.DOCUMENTS_DIR)
            chunker = DocumentChunker(
                chunk_size=config.CHUNK_SIZE,
                overlap_size=config.OVERLAP_SIZE
            )
            embedding_gen = EmbeddingGenerator(
                model_name=config.EMBEDDING_MODEL,
                api_key=config.GEMINI_API_KEY
            )
            db_manager = create_database_manager(
                config.CONNECTION_STRING,
                config.TABLE_NAME
            )
            
            logger.info("✅ Components initialized")
            
            # =================================================================
            # STAGE 1: Document Loading (skip if skip_conversion AND skip_indexing)
            # =================================================================
            
            if skip_indexing:
                logger.info("⏩ Skipping entire indexing pipeline (skip_indexing=True)")
                task.status = IndexingStatus.COMPLETED
                task.end_time = datetime.now()
                self._add_to_history(task)
                return
            
            task.current_stage = "loading"
            task.current_stage_name = "Loading Documents"
            
            logger.info("📂 Loading markdown documents...")
            
            # Load documents
            documents = doc_loader.load_documents()
            
            if not documents:
                logger.warning("⚠️ No documents found to index")
                task.warnings.append("No documents found in directory")
                task.status = IndexingStatus.COMPLETED
                task.end_time = datetime.now()
                self._add_to_history(task)
                return
            
            task.total_files = len(documents)
            logger.info(f"📄 Loaded {len(documents)} documents")
            
            # =================================================================
            # STAGE 2: Chunking
            # =================================================================
            
            task.current_stage = "chunking"
            task.current_stage_name = "Chunking Documents"
            
            logger.info("✂️ Chunking documents...")
            
            all_chunks = []
            for i, doc in enumerate(documents, 1):
                if task.cancelled:
                    logger.info("⚠️ Indexing cancelled by user")
                    task.status = IndexingStatus.FAILED
                    task.errors.append("Cancelled by user")
                    break
                
                task.current_file = doc.metadata.get('file_name', 'unknown')
                logger.info(f"[{i}/{len(documents)}] Chunking: {task.current_file}")
                
                try:
                    chunks = chunker.chunk_document(doc)
                    all_chunks.extend(chunks)
                    task.processed_files += 1
                    logger.info(f"  ✅ Created {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"  ❌ Chunking failed: {e}")
                    task.failed_files += 1
                    task.errors.append(f"{task.current_file}: Chunking failed - {str(e)}")
            
            task.total_chunks = len(all_chunks)
            logger.info(f"📊 Total chunks created: {len(all_chunks)}")
            
            if not all_chunks:
                logger.warning("⚠️ No chunks created")
                task.warnings.append("No chunks created from documents")
                task.status = IndexingStatus.COMPLETED
                task.end_time = datetime.now()
                self._add_to_history(task)
                return
            
            # =================================================================
            # STAGE 3: Embedding Generation
            # =================================================================
            
            task.current_stage = "embedding"
            task.current_stage_name = "Generating Embeddings"
            
            logger.info("🧮 Generating embeddings...")
            
            # Process in batches
            actual_batch_size = batch_size or config.BATCH_SIZE
            total_batches = (len(all_chunks) + actual_batch_size - 1) // actual_batch_size
            
            embedded_chunks = []
            for batch_idx in range(0, len(all_chunks), actual_batch_size):
                if task.cancelled:
                    break
                
                batch = all_chunks[batch_idx:batch_idx + actual_batch_size]
                batch_num = (batch_idx // actual_batch_size) + 1
                
                logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                
                try:
                    batch_with_embeddings = embedding_gen.generate_embeddings(batch)
                    embedded_chunks.extend(batch_with_embeddings)
                    task.processed_chunks += len(batch)
                    logger.info(f"  ✅ Generated {len(batch)} embeddings")
                except Exception as e:
                    logger.error(f"  ❌ Embedding generation failed: {e}")
                    task.errors.append(f"Batch {batch_num}: Embedding failed - {str(e)}")
            
            logger.info(f"✅ Generated embeddings for {len(embedded_chunks)} chunks")
            
            # =================================================================
            # STAGE 4: Database Save
            # =================================================================
            
            task.current_stage = "saving"
            task.current_stage_name = "Saving to Database"
            
            logger.info("💾 Saving to database...")
            
            # Delete existing if requested
            if delete_existing:
                logger.warning("⚠️ Deleting existing records...")
                # Implement deletion if needed
            
            # Save to database
            try:
                saved_count = db_manager.insert_chunks(embedded_chunks)
                logger.info(f"✅ Saved {saved_count} chunks to database")
                
                task.statistics = {
                    "documents_loaded": len(documents),
                    "chunks_created": len(all_chunks),
                    "chunks_saved": saved_count,
                    "success_rate": saved_count / len(all_chunks) if all_chunks else 0,
                }
            except Exception as e:
                logger.error(f"❌ Database save failed: {e}")
                task.errors.append(f"Database save failed: {str(e)}")
                task.status = IndexingStatus.FAILED
                task.end_time = datetime.now()
                self._add_to_history(task)
                return
            
            # =================================================================
            # COMPLETE
            # =================================================================
            
            task.status = IndexingStatus.COMPLETED
            task.end_time = datetime.now()
            task.current_file = None
            task.current_stage = "completed"
            task.current_stage_name = "Completed"
            
            # Calculate duration
            duration = (task.end_time - task.start_time).total_seconds()
            
            logger.info(f"✅ REAL indexing pipeline completed: {task.task_id}")
            logger.info(f"   📊 Files: {task.processed_files}/{task.total_files}")
            logger.info(f"   🧩 Chunks: {task.processed_chunks}/{task.total_chunks}")
            logger.info(f"   ⏱️ Duration: {duration:.1f}s")
            
            # Add to history
            self._add_to_history(task)
            
        except asyncio.CancelledError:
            logger.info(f"⚠️ Indexing task cancelled: {task.task_id}")
            task.status = IndexingStatus.FAILED
            task.end_time = datetime.now()
            task.errors.append("Task cancelled")
            self._add_to_history(task)
            
        except Exception as e:
            logger.error(f"❌ Indexing failed for task {task.task_id}: {e}", exc_info=True)
            task.status = IndexingStatus.FAILED
            task.end_time = datetime.now()
            task.errors.append(f"Indexing error: {str(e)}")
            self._add_to_history(task)
    
    def _add_to_history(self, task: IndexingTaskState):
        """Add completed task to history"""
        history_item = IndexingHistoryItem(
            task_id=task.task_id,
            mode=task.mode,
            status=task.status,
            start_time=task.start_time,
            end_time=task.end_time,
            total_files=task.total_files,
            processed_files=task.processed_files,
            failed_files=task.failed_files,
            total_chunks=task.total_chunks,
            processed_chunks=task.processed_chunks,
        )
        
        self._history.append(history_item)
        
        # Keep only last 50 items
        if len(self._history) > 50:
            self._history = self._history[-50:]
        
        logger.info(f"📝 Added task to history: {task.task_id}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel running indexing task"""
        task = await self.get_task(task_id)
        if not task:
            return False
        
        if task.status != IndexingStatus.RUNNING:
            logger.warning(f"⚠️ Cannot cancel task {task_id}: not running")
            return False
        
        task.cancelled = True
        logger.info(f"⚠️ Cancelling indexing task: {task_id}")
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
    
    async def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks"""
        return [
            {
                "task_id": task_id,
                "status": task.status.value,
                "mode": task.mode.value,
                "progress_percentage": task.get_progress().progress_percentage,
            }
            for task_id, task in self._tasks.items()
        ]
    
    async def clear_completed_tasks(self):
        """Clear completed/failed tasks"""
        async with self._lock:
            self._tasks = {
                task_id: task
                for task_id, task in self._tasks.items()
                if task.status == IndexingStatus.RUNNING
            }
            logger.info("🧹 Cleared completed tasks")
    
    def get_active_tasks_count(self) -> int:
        """Get number of active indexing tasks"""
        return sum(
            1 for task in self._tasks.values()
            if task.status == IndexingStatus.RUNNING
        )
    
    async def get_history(self, limit: int = 10) -> List[IndexingHistoryItem]:
        """Get indexing history"""
        return self._history[-limit:]


# Singleton instance
_indexing_service: Optional[IndexingService] = None


def get_indexing_service() -> IndexingService:
    """Get or create indexing service singleton"""
    global _indexing_service
    
    if _indexing_service is None:
        _indexing_service = IndexingService()
    
    return _indexing_service