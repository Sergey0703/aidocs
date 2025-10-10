#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# api/modules/vehicles/routes/documents.py
# Document management and linking operations

import logging
from fastapi import APIRouter, HTTPException
from typing import List

from ..models.schemas import (
    LinkDocumentRequest,
    UnlinkDocumentRequest,
    LinkDocumentResponse,
    UnlinkDocumentResponse,
    UnassignedDocumentsResponse,
    AnalyzeDocumentsResponse,
    GroupedDocumentsByVRN,
    DocumentRegistryItem,
    VehicleResponse,
    DocumentStatistics,
    ErrorResponse,
)
from ..services.vehicle_service import get_vehicle_service
from ..services.document_registry_service import get_document_registry_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/{vehicle_id}/documents/link", response_model=LinkDocumentResponse, responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def link_document_to_vehicle(vehicle_id: str, request: LinkDocumentRequest):
    """
    Link document to vehicle.
    
    **Process:**
    1. Validates vehicle exists
    2. Validates document exists
    3. Links document to vehicle
    4. Updates document status to 'assigned'
    
    **Example:**
    ```json
    {
      "registry_id": "uuid-document-123"
    }
    ```
    
    **Use cases:**
    - Assign uploaded document to vehicle
    - Link insurance certificate
    - Associate service records
    """
    try:
        vehicle_service = get_vehicle_service()
        registry_service = get_document_registry_service()
        
        # Validate vehicle exists
        vehicle = await vehicle_service.get_by_id(vehicle_id)
        if not vehicle:
            logger.warning(f"Vehicle not found for linking: {vehicle_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Vehicle not found: {vehicle_id}"
            )
        
        # Validate document exists
        document = await registry_service.get_by_id(request.registry_id)
        if not document:
            logger.warning(f"Document not found for linking: {request.registry_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {request.registry_id}"
            )
        
        # Check if document is already linked to another vehicle
        if document.get('vehicle_id') and document['vehicle_id'] != vehicle_id:
            existing_vehicle = await vehicle_service.get_by_id(document['vehicle_id'])
            vrn = existing_vehicle.get('registration_number', 'another vehicle') if existing_vehicle else 'another vehicle'
            raise HTTPException(
                status_code=400,
                detail=f"Document is already linked to {vrn}. Unlink it first."
            )
        
        # Link document to vehicle
        success = await registry_service.link_to_vehicle(request.registry_id, vehicle_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to link document to vehicle"
            )
        
        vrn = vehicle.get('registration_number', vehicle_id)
        filename = document.get('raw_file_path', '').split('/')[-1] or request.registry_id
        
        logger.info(f"✅ Linked document {request.registry_id} to vehicle {vehicle_id} ({vrn})")
        
        return LinkDocumentResponse(
            success=True,
            message=f"Document '{filename}' successfully linked to vehicle '{vrn}'",
            document_id=request.registry_id,
            vehicle_id=vehicle_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to link document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to link document: {str(e)}"
        )


@router.post("/{vehicle_id}/documents/unlink", response_model=UnlinkDocumentResponse, responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def unlink_document_from_vehicle(vehicle_id: str, request: UnlinkDocumentRequest):
    """
    Unlink document from vehicle.
    
    **Process:**
    1. Validates document exists
    2. Checks document is linked to this vehicle
    3. Unlinks document (sets vehicle_id to NULL)
    4. Updates document status to 'unassigned'
    
    **Example:**
    ```json
    {
      "registry_id": "uuid-document-123"
    }
    ```
    
    **Use cases:**
    - Correct incorrect linking
    - Remove outdated documents
    - Reassign document to different vehicle
    """
    try:
        registry_service = get_document_registry_service()
        
        # Validate document exists
        document = await registry_service.get_by_id(request.registry_id)
        if not document:
            logger.warning(f"Document not found for unlinking: {request.registry_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {request.registry_id}"
            )
        
        # Check if document is actually linked to this vehicle
        if document.get('vehicle_id') != vehicle_id:
            if document.get('vehicle_id') is None:
                raise HTTPException(
                    status_code=400,
                    detail="Document is not linked to any vehicle"
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Document is linked to a different vehicle"
                )
        
        # Unlink document
        success = await registry_service.unlink_from_vehicle(request.registry_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to unlink document from vehicle"
            )
        
        filename = document.get('raw_file_path', '').split('/')[-1] or request.registry_id
        
        logger.info(f"✅ Unlinked document {request.registry_id} from vehicle {vehicle_id}")
        
        return UnlinkDocumentResponse(
            success=True,
            message=f"Document '{filename}' successfully unlinked from vehicle",
            document_id=request.registry_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unlink document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unlink document: {str(e)}"
        )


@router.get("/documents/unassigned", response_model=UnassignedDocumentsResponse, responses={500: {"model": ErrorResponse}})
async def get_unassigned_documents():
    """
    Get all documents not linked to any vehicle.
    
    **Returns:**
    - Documents with status 'unassigned'
    - Documents without vehicle_id
    - Sorted by upload date (newest first)
    
    **Use cases:**
    - Review pending documents
    - Find documents to link
    - Cleanup unassigned documents
    """
    try:
        registry_service = get_document_registry_service()
        
        # Get unassigned documents
        documents = await registry_service.get_unassigned(limit=1000)
        
        # Convert to response models
        document_items = [
            DocumentRegistryItem(
                id=str(doc['id']),
                vehicle_id=None,
                raw_file_path=doc.get('raw_file_path'),
                markdown_file_path=doc.get('markdown_file_path'),
                document_type=doc.get('document_type'),
                status=doc['status'],
                extracted_data=doc.get('extracted_data', {}),
                uploaded_at=doc['uploaded_at'],
                updated_at=doc['updated_at'],
                filename=doc.get('raw_file_path', '').split('/')[-1] if doc.get('raw_file_path') else None,
                is_indexed=(doc['status'] == 'processed')
            )
            for doc in documents
        ]
        
        logger.info(f"Retrieved {len(document_items)} unassigned documents")
        
        return UnassignedDocumentsResponse(
            documents=document_items,
            total=len(document_items)
        )
        
    except Exception as e:
        logger.error(f"Failed to get unassigned documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve unassigned documents: {str(e)}"
        )


@router.get("/documents/analyze", response_model=AnalyzeDocumentsResponse, responses={500: {"model": ErrorResponse}})
async def analyze_documents():
    """
    Analyze documents and group by extracted VRN.
    
    **Process:**
    1. Gets all unassigned documents
    2. Groups by extracted VRN (from extracted_data.vrn)
    3. Checks if vehicles exist for each VRN
    4. Returns grouped and truly unassigned documents
    
    **Returns:**
    - **grouped**: Documents grouped by VRN
      - If vehicle exists: includes vehicle details
      - If vehicle doesn't exist: suggests creation
    - **unassigned**: Documents without VRN extraction
    
    **Use cases:**
    - Document Manager page
    - Bulk linking workflow
    - Auto-create vehicles from documents
    
    **Example response:**
    ```json
    {
      "grouped": [
        {
          "vrn": "191-D-12345",
          "vehicle_exists": true,
          "vehicle_details": {...},
          "documents": [...]
        },
        {
          "vrn": "241-KY-999",
          "vehicle_exists": false,
          "suggested_make": "Toyota",
          "suggested_model": "Yaris",
          "documents": [...]
        }
      ],
      "unassigned": [...],
      "vehicles_needing_creation": 1
    }
    ```
    """
    try:
        registry_service = get_document_registry_service()
        vehicle_service = get_vehicle_service()
        
        # Group documents by VRN
        grouped_by_vrn = await registry_service.group_by_extracted_vrn()
        
        # Get unassigned documents (those without VRN)
        all_unassigned = await registry_service.get_unassigned(limit=1000)
        
        # Filter out documents that have VRN (they're in grouped_by_vrn)
        truly_unassigned = [
            doc for doc in all_unassigned
            if not doc.get('extracted_data', {}).get('vrn')
        ]
        
        # Process grouped documents
        grouped_results = []
        vehicles_needing_creation = 0
        
        for vrn, documents in grouped_by_vrn.items():
            # Check if vehicle exists
            vehicle = await vehicle_service.get_by_registration(vrn)
            
            vehicle_exists = vehicle is not None
            if not vehicle_exists:
                vehicles_needing_creation += 1
            
            # Extract suggested make/model from documents
            suggested_make = None
            suggested_model = None
            
            if not vehicle_exists:
                for doc in documents:
                    extracted = doc.get('extracted_data', {})
                    if not suggested_make and extracted.get('make'):
                        suggested_make = extracted['make']
                    if not suggested_model and extracted.get('model'):
                        suggested_model = extracted['model']
                    if suggested_make and suggested_model:
                        break
            
            # Convert documents to response models
            document_items = [
                DocumentRegistryItem(
                    id=str(doc['id']),
                    vehicle_id=str(doc['vehicle_id']) if doc.get('vehicle_id') else None,
                    raw_file_path=doc.get('raw_file_path'),
                    markdown_file_path=doc.get('markdown_file_path'),
                    document_type=doc.get('document_type'),
                    status=doc['status'],
                    extracted_data=doc.get('extracted_data', {}),
                    uploaded_at=doc['uploaded_at'],
                    updated_at=doc['updated_at'],
                    filename=doc.get('raw_file_path', '').split('/')[-1] if doc.get('raw_file_path') else None,
                    is_indexed=(doc['status'] == 'processed')
                )
                for doc in documents
            ]
            
            grouped_results.append(GroupedDocumentsByVRN(
                vrn=vrn,
                vehicle_exists=vehicle_exists,
                vehicle_details=VehicleResponse(**vehicle) if vehicle else None,
                suggested_make=suggested_make,
                suggested_model=suggested_model,
                documents=document_items
            ))
        
        # Convert unassigned to response models
        unassigned_items = [
            DocumentRegistryItem(
                id=str(doc['id']),
                vehicle_id=None,
                raw_file_path=doc.get('raw_file_path'),
                markdown_file_path=doc.get('markdown_file_path'),
                document_type=doc.get('document_type'),
                status=doc['status'],
                extracted_data=doc.get('extracted_data', {}),
                uploaded_at=doc['uploaded_at'],
                updated_at=doc['updated_at'],
                filename=doc.get('raw_file_path', '').split('/')[-1] if doc.get('raw_file_path') else None,
                is_indexed=(doc['status'] == 'processed')
            )
            for doc in truly_unassigned
        ]
        
        logger.info(f"📊 Document analysis: {len(grouped_results)} groups, {len(unassigned_items)} unassigned, {vehicles_needing_creation} vehicles need creation")
        
        return AnalyzeDocumentsResponse(
            grouped=grouped_results,
            unassigned=unassigned_items,
            total_grouped=len(grouped_results),
            total_unassigned=len(unassigned_items),
            vehicles_with_documents=sum(1 for g in grouped_results if g.vehicle_exists),
            vehicles_needing_creation=vehicles_needing_creation
        )
        
    except Exception as e:
        logger.error(f"Failed to analyze documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze documents: {str(e)}"
        )


@router.get("/documents/stats", response_model=DocumentStatistics, responses={500: {"model": ErrorResponse}})
async def get_document_statistics():
    """
    Get comprehensive document registry statistics.
    
    **Returns:**
    - Total documents
    - Documents by status
    - Documents by type
    - Assigned vs unassigned
    
    **Statistics include:**
    - `total_documents` - Total in registry
    - `assigned_documents` - Linked to vehicles
    - `unassigned_documents` - Not linked
    - `pending_indexing` - Awaiting indexing
    - `processed_documents` - Fully processed
    - `failed_documents` - Processing failed
    - `documents_by_type` - Breakdown by document type
    
    **Use cases:**
    - Monitoring dashboard
    - Processing pipeline health
    - Data quality metrics
    """
    try:
        registry_service = get_document_registry_service()
        
        stats = await registry_service.get_statistics()
        
        # Map stats to response model
        return DocumentStatistics(
            total_documents=stats.get('total_documents', 0),
            assigned_documents=stats.get('assigned', 0),
            unassigned_documents=stats.get('unassigned', 0),
            pending_indexing=stats.get('by_status', {}).get('pending_indexing', 0),
            processed_documents=stats.get('by_status', {}).get('processed', 0),
            failed_documents=stats.get('by_status', {}).get('failed', 0),
            documents_by_type=stats.get('by_type', {})
        )
        
    except Exception as e:
        logger.error(f"Failed to get document statistics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )