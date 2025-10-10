#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# api/modules/document_inbox/routes/inbox.py
# Document Inbox routes - batch operations, vehicle creation, search

import logging
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from api.modules.vehicles.services.vehicle_service import get_vehicle_service
from api.modules.vehicles.services.document_registry_service import get_document_registry_service
from api.modules.vehicles.models.schemas import VehicleResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class LinkBatchRequest(BaseModel):
    """Request to link multiple documents to a vehicle"""
    registry_ids: List[str] = Field(..., min_items=1, description="List of document registry UUIDs")


class LinkBatchResponse(BaseModel):
    """Response after batch linking"""
    success: bool
    message: str
    vehicle_id: str
    linked_count: int
    failed_ids: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.now)


class CreateVehicleAndLinkRequest(BaseModel):
    """Request to create vehicle and link documents"""
    registration_number: str = Field(..., min_length=1, max_length=50)
    make: Optional[str] = None
    model: Optional[str] = None
    vin_number: Optional[str] = None
    document_ids: List[str] = Field(..., min_items=1, description="Document registry IDs to link")


class CreateVehicleAndLinkResponse(BaseModel):
    """Response after creating vehicle and linking documents"""
    success: bool
    message: str
    vehicle: VehicleResponse
    linked_count: int
    failed_ids: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.now)


class VehicleSearchResult(BaseModel):
    """Vehicle search result for dropdown"""
    id: str
    registration_number: str
    make: Optional[str] = None
    model: Optional[str] = None
    status: str


class VehicleSearchResponse(BaseModel):
    """Response with vehicle search results"""
    results: List[VehicleSearchResult]
    total: int
    timestamp: datetime = Field(default_factory=datetime.now)


class UnlinkBatchRequest(BaseModel):
    """Request to unlink multiple documents"""
    registry_ids: List[str] = Field(..., min_items=1, description="List of document registry UUIDs")


class UnlinkBatchResponse(BaseModel):
    """Response after batch unlinking"""
    success: bool
    message: str
    unlinked_count: int
    failed_ids: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# BATCH LINK DOCUMENTS TO VEHICLE
# ============================================================================

@router.post("/link-batch", response_model=LinkBatchResponse)
async def link_documents_batch(vehicle_id: str, request: LinkBatchRequest):
    """
    Link multiple documents to a vehicle in one operation.
    """
    try:
        vehicle_service = get_vehicle_service()
        registry_service = get_document_registry_service()
        
        # Validate vehicle exists
        vehicle = await vehicle_service.get_by_id(vehicle_id)
        if not vehicle:
            logger.warning(f"Vehicle not found for batch linking: {vehicle_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Vehicle not found: {vehicle_id}"
            )
        
        vrn = vehicle.get('registration_number', vehicle_id)
        logger.info(f"📦 Batch linking {len(request.registry_ids)} documents to {vrn}")
        
        # Link each document
        linked_count = 0
        failed_ids = []
        
        for registry_id in request.registry_ids:
            try:
                # Validate document exists
                document = await registry_service.get_by_id(registry_id)
                if not document:
                    logger.warning(f"Document not found: {registry_id}")
                    failed_ids.append(registry_id)
                    continue
                
                # Check if already linked to another vehicle
                if document.get('vehicle_id') and document['vehicle_id'] != vehicle_id:
                    existing_vehicle = await vehicle_service.get_by_id(document['vehicle_id'])
                    existing_vrn = existing_vehicle.get('registration_number', 'another vehicle') if existing_vehicle else 'another vehicle'
                    logger.warning(f"Document {registry_id} already linked to {existing_vrn}")
                    failed_ids.append(registry_id)
                    continue
                
                # Link document
                success = await registry_service.link_to_vehicle(registry_id, vehicle_id)
                
                if success:
                    linked_count += 1
                    logger.debug(f"  ✅ Linked: {registry_id}")
                else:
                    failed_ids.append(registry_id)
                    logger.warning(f"  ❌ Failed to link: {registry_id}")
                    
            except Exception as e:
                logger.error(f"Error linking document {registry_id}: {e}")
                failed_ids.append(registry_id)
                continue
        
        # Generate response message
        if linked_count == len(request.registry_ids):
            message = f"Successfully linked all {linked_count} documents to vehicle '{vrn}'"
            logger.info(f"✅ Batch link complete: {linked_count}/{len(request.registry_ids)} successful")
        elif linked_count > 0:
            message = f"Linked {linked_count}/{len(request.registry_ids)} documents to vehicle '{vrn}'. {len(failed_ids)} failed."
            logger.warning(f"⚠️ Partial batch link: {linked_count}/{len(request.registry_ids)} successful")
        else:
            message = f"Failed to link any documents to vehicle '{vrn}'"
            logger.error(f"❌ Batch link failed: 0/{len(request.registry_ids)} successful")
        
        return LinkBatchResponse(
            success=linked_count > 0,
            message=message,
            vehicle_id=vehicle_id,
            linked_count=linked_count,
            failed_ids=failed_ids
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch linking failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch linking operation failed: {str(e)}"
        )


# ============================================================================
# BATCH UNLINK DOCUMENTS
# ============================================================================

@router.post("/unlink-batch", response_model=UnlinkBatchResponse)
async def unlink_documents_batch(request: UnlinkBatchRequest):
    """
    Unlink multiple documents from their vehicles.
    """
    try:
        registry_service = get_document_registry_service()
        
        logger.info(f"📦 Batch unlinking {len(request.registry_ids)} documents")
        
        unlinked_count = 0
        failed_ids = []
        
        for registry_id in request.registry_ids:
            try:
                # Validate document exists
                document = await registry_service.get_by_id(registry_id)
                if not document:
                    logger.warning(f"Document not found: {registry_id}")
                    failed_ids.append(registry_id)
                    continue
                
                # Check if document is actually linked
                if not document.get('vehicle_id'):
                    logger.warning(f"Document {registry_id} is not linked to any vehicle")
                    failed_ids.append(registry_id)
                    continue
                
                # Unlink document
                success = await registry_service.unlink_from_vehicle(registry_id)
                
                if success:
                    unlinked_count += 1
                    logger.debug(f"  ✅ Unlinked: {registry_id}")
                else:
                    failed_ids.append(registry_id)
                    logger.warning(f"  ❌ Failed to unlink: {registry_id}")
                    
            except Exception as e:
                logger.error(f"Error unlinking document {registry_id}: {e}")
                failed_ids.append(registry_id)
                continue
        
        # Generate response message
        if unlinked_count == len(request.registry_ids):
            message = f"Successfully unlinked all {unlinked_count} documents"
            logger.info(f"✅ Batch unlink complete: {unlinked_count}/{len(request.registry_ids)} successful")
        elif unlinked_count > 0:
            message = f"Unlinked {unlinked_count}/{len(request.registry_ids)} documents. {len(failed_ids)} failed."
            logger.warning(f"⚠️ Partial batch unlink: {unlinked_count}/{len(request.registry_ids)} successful")
        else:
            message = f"Failed to unlink any documents"
            logger.error(f"❌ Batch unlink failed: 0/{len(request.registry_ids)} successful")
        
        return UnlinkBatchResponse(
            success=unlinked_count > 0,
            message=message,
            unlinked_count=unlinked_count,
            failed_ids=failed_ids
        )
        
    except Exception as e:
        logger.error(f"Batch unlinking failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch unlinking operation failed: {str(e)}"
        )


# ============================================================================
# CREATE VEHICLE AND LINK DOCUMENTS
# ============================================================================

@router.post("/create-vehicle-and-link", response_model=CreateVehicleAndLinkResponse)
async def create_vehicle_and_link_documents(request: CreateVehicleAndLinkRequest):
    """
    Create new vehicle and link documents in one operation.
    """
    try:
        vehicle_service = get_vehicle_service()
        registry_service = get_document_registry_service()
        
        logger.info(f"🚗 Creating vehicle {request.registration_number} and linking {len(request.document_ids)} documents")
        
        # Step 1: Create vehicle
        try:
            vehicle_id = await vehicle_service.create_vehicle(
                registration_number=request.registration_number,
                vin_number=request.vin_number,
                make=request.make,
                model=request.model,
                status='active'
            )
            
            logger.info(f"✅ Vehicle created: {vehicle_id} ({request.registration_number})")
            
        except ValueError as e:
            # Duplicate VRN or VIN
            logger.warning(f"Vehicle creation failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        
        # Step 2: Get created vehicle
        vehicle = await vehicle_service.get_by_id(vehicle_id)
        if not vehicle:
            raise HTTPException(
                status_code=500,
                detail="Vehicle created but could not be retrieved"
            )
        
        # Step 3: Link documents
        linked_count = 0
        failed_ids = []
        
        for doc_id in request.document_ids:
            try:
                # Validate document exists
                document = await registry_service.get_by_id(doc_id)
                if not document:
                    logger.warning(f"Document not found: {doc_id}")
                    failed_ids.append(doc_id)
                    continue
                
                # Check if already linked to another vehicle
                if document.get('vehicle_id'):
                    existing_vehicle = await vehicle_service.get_by_id(document['vehicle_id'])
                    existing_vrn = existing_vehicle.get('registration_number', 'another vehicle') if existing_vehicle else 'another vehicle'
                    logger.warning(f"Document {doc_id} already linked to {existing_vrn}")
                    failed_ids.append(doc_id)
                    continue
                
                # Link document to new vehicle
                success = await registry_service.link_to_vehicle(doc_id, vehicle_id)
                
                if success:
                    linked_count += 1
                    logger.debug(f"  ✅ Linked: {doc_id}")
                else:
                    failed_ids.append(doc_id)
                    logger.warning(f"  ❌ Failed to link: {doc_id}")
                    
            except Exception as e:
                logger.error(f"Error linking document {doc_id}: {e}")
                failed_ids.append(doc_id)
                continue
        
        # Generate response message
        if linked_count == len(request.document_ids):
            message = f"Successfully created vehicle '{request.registration_number}' and linked all {linked_count} documents"
            logger.info(f"✅ Create and link complete: vehicle created, {linked_count}/{len(request.document_ids)} docs linked")
        elif linked_count > 0:
            message = f"Vehicle '{request.registration_number}' created. Linked {linked_count}/{len(request.document_ids)} documents. {len(failed_ids)} failed."
            logger.warning(f"⚠️ Partial create and link: vehicle created, {linked_count}/{len(request.document_ids)} docs linked")
        else:
            message = f"Vehicle '{request.registration_number}' created but failed to link any documents"
            logger.warning(f"⚠️ Create and link: vehicle created, 0/{len(request.document_ids)} docs linked")
        
        return CreateVehicleAndLinkResponse(
            success=True,
            message=message,
            vehicle=VehicleResponse(**vehicle),
            linked_count=linked_count,
            failed_ids=failed_ids
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create vehicle and link failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Create and link operation failed: {str(e)}"
        )


# ============================================================================
# SEARCH VEHICLES FOR DROPDOWN
# ============================================================================

@router.get("/search-vehicles", response_model=VehicleSearchResponse)
async def search_vehicles_for_inbox(query: str = "", limit: int = 10):
    """
    Search vehicles by registration number for dropdown autocomplete.
    """
    try:
        vehicle_service = get_vehicle_service()
        
        logger.info(f"🔍 Searching vehicles for inbox: '{query}'")
        
        # Get all vehicles (we'll filter manually for partial match)
        all_vehicles, total = await vehicle_service.get_all(
            status='active',
            limit=1000,
            offset=0
        )
        
        # Filter by query (case-insensitive partial match)
        if query:
            query_lower = query.lower()
            filtered_vehicles = [
                v for v in all_vehicles
                if query_lower in v.get('registration_number', '').lower()
            ]
        else:
            filtered_vehicles = all_vehicles
        
        # Sort by registration number
        filtered_vehicles.sort(key=lambda v: v.get('registration_number', ''))
        
        # Limit results
        limited_vehicles = filtered_vehicles[:limit]
        
        # Convert to response format
        results = [
            VehicleSearchResult(
                id=str(v['id']),
                registration_number=v['registration_number'],
                make=v.get('make'),
                model=v.get('model'),
                status=v.get('status', 'active')
            )
            for v in limited_vehicles
        ]
        
        logger.info(f"✅ Found {len(results)} vehicles matching '{query}'")
        
        return VehicleSearchResponse(
            results=results,
            total=len(results)
        )
        
    except Exception as e:
        logger.error(f"Vehicle search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Vehicle search failed: {str(e)}"
        )