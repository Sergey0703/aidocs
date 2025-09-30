# api/models/schemas.py
# Pydantic models for API request/response validation

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class SearchRequest(BaseModel):
    """Request model for search endpoint"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    max_results: Optional[int] = Field(default=20, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Similarity threshold")
    

class EntityResult(BaseModel):
    """Entity extraction result"""
    entity: str
    confidence: float
    method: str
    alternatives: List[str] = []
    metadata: Dict[str, Any] = {}


class QueryRewriteResult(BaseModel):
    """Query rewriting result"""
    original_query: str
    rewrites: List[str]
    method: str
    confidence: float
    metadata: Dict[str, Any] = {}


class DocumentResult(BaseModel):
    """Single document result"""
    filename: str
    content: str
    full_content: str
    similarity_score: float
    source_method: str
    document_id: str = ""
    chunk_index: int = 0
    metadata: Dict[str, Any] = {}


class PerformanceMetrics(BaseModel):
    """Performance metrics for search pipeline"""
    total_time: float
    extraction_time: float
    rewrite_time: float
    retrieval_time: float
    fusion_time: float
    answer_time: float
    pipeline_efficiency: Dict[str, float]


class SearchResponse(BaseModel):
    """Response model for search endpoint"""
    success: bool
    query: str
    entity_result: EntityResult
    rewrite_result: QueryRewriteResult
    results: List[DocumentResult]
    answer: str
    total_results: int
    performance_metrics: PerformanceMetrics
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    error_type: str
    timestamp: datetime = Field(default_factory=datetime.now)


class SystemStatus(BaseModel):
    """System status response"""
    status: str
    components: Dict[str, bool]
    database: Dict[str, Any]
    embedding: Dict[str, Any]
    hybrid_enabled: bool
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)