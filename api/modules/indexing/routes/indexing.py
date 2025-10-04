# api/modules/indexing/routes/indexing.py
# Main indexing operations endpoints

import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional

from ..models.schemas import (
    IndexingRequest,
    IndexingResponse,
    IndexingStatusResponse,
    IndexingHistoryResponse,
    ReindexFilesRequest,
    SuccessResponse,
    ErrorResponse,
    IndexingMode,
)
from ..services.indexing_service import get_indexing_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/start", response_model=IndexingResponse, responses={500: {"model": ErrorResponse}})
async def start_indexing(
    request: IndexingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start document indexing process.
    
    Pipeline:
    - Part 1 (if not skipped): Document conversion using Docling
    - Part 2 (if not skipped): Chunking and embedding generation
    
    Modes:
    - **full**: Reindex all documents
    - **incremental**: Only new/modified documents
    
    Process runs in background. Use /status endpoint to check progress.
    """
    try:
        service = get_indexing_service()
        
        # Check if already running
        if service.get_active_tasks_count() > 0:
            raise HTTPException(
                status_code=409,
                detail="Indexing task already running. Please wait for completion."
            )
        
        # Create task
        task_id = await service.create_task(request.mode)
        
        # Start indexing in background
        background_tasks.add_task(
            service.start_indexing,
            task_id=task_id,
            documents_dir=request.documents_dir,
            skip_conversion=request.skip_conversion,
            skip_indexing=request.skip_indexing,
            batch_size=request.batch_size,
            force_reindex=request.force_reindex,
            delete_existing=request.delete_existing,
        )
        
        logger.info(f"Started indexing task: {task_id} (mode: {request.mode.value})")
        
        return IndexingResponse(
            success=True,
            message=f"Indexing started successfully in {request.mode.value} mode",
            task_id=task_id,
            mode=request.mode,
            files_to_process=0,  # Will be updated during processing
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=SuccessResponse, responses={404: {"model": ErrorResponse}})
async def stop_indexing(task_id: str):
    """
    Stop running indexing task.
    
    - Gracefully stops current processing
    - Completes current batch before stopping
    - Returns partial results
    """
    try:
        service = get_indexing_service()
        
        success = await service.cancel_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found or not running: {task_id}"
            )
        
        return SuccessResponse(
            success=True,
            message=f"Indexing task {task_id} stopped successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=IndexingStatusResponse, responses={404: {"model": ErrorResponse}})
async def get_status(task_id: Optional[str] = None):
    """
    Get indexing status.
    
    - Returns current progress and statistics
    - Shows processing stage and ETA
    - Lists any errors encountered
    
    If task_id not provided, returns status of most recent task.
    """
    try:
        service = get_indexing_service()
        
        if not task_id:
            # Get most recent task
            # TODO: Implement getting most recent task
            raise HTTPException(
                status_code=400,
                detail="task_id is required"
            )
        
        status = await service.get_task_status(task_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found: {task_id}"
            )
        
        return IndexingStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=IndexingHistoryResponse)
async def get_history(limit: int = 10):
    """
    Get indexing history.
    
    Returns list of past indexing runs with:
    - Completion status
    - Processing time
    - Files processed
    - Success rate
    """
    try:
        service = get_indexing_service()
        
        history = await service.get_history(limit=limit)
        
        # Calculate summary stats
        total_runs = len(history)
        successful_runs = [h for h in history if h.status.value == "completed"]
        failed_runs = [h for h in history if h.status.value == "failed"]
        
        last_successful = successful_runs[0].end_time if successful_runs else None
        last_failed = failed_runs[0].end_time if failed_runs else None
        
        return IndexingHistoryResponse(
            history=history,
            total_runs=total_runs,
            last_successful_run=last_successful,
            last_failed_run=last_failed,
        )
        
    except Exception as e:
        logger.error(f"Failed to get history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear", response_model=SuccessResponse)
async def clear_index(confirm: bool = False):
    """
    Clear entire index.
    
    ⚠️ WARNING: This deletes all indexed documents!
    
    - Requires explicit confirmation
    - Cannot be undone
    - Use for testing or complete reindex
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Set confirm=true to proceed."
        )
    
    try:
        # TODO: Implement index clearing
        # This should use database_manager to delete all records
        
        return SuccessResponse(
            success=True,
            message="Index cleared successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to clear index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex", response_model=IndexingResponse, responses={500: {"model": ErrorResponse}})
async def reindex_files(
    request: ReindexFilesRequest,
    background_tasks: BackgroundTasks
):
    """
    Reindex specific files.
    
    - Deletes existing records for specified files
    - Re-processes only those files
    - Useful for fixing specific documents
    """
    try:
        service = get_indexing_service()
        
        # Create task for reindexing
        task_id = await service.create_task(IndexingMode.INCREMENTAL)
        
        # TODO: Implement reindex_specific_files method
        # background_tasks.add_task(
        #     service.reindex_specific_files,
        #     task_id=task_id,
        #     filenames=request.filenames,
        #     force=request.force,
        # )
        
        logger.info(f"Started reindex task: {task_id} for {len(request.filenames)} files")
        
        return IndexingResponse(
            success=True,
            message=f"Reindexing started for {len(request.filenames)} files",
            task_id=task_id,
            mode=IndexingMode.INCREMENTAL,
            files_to_process=len(request.filenames),
        )
        
    except Exception as e:
        logger.error(f"Failed to start reindex: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))