# api/models/schemas.py
# Pydantic models for API request/response validation

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class RerankMode(str, Enum):
    """Re-ranking modes"""
    SMART = "smart"  # Auto-skip when not needed (default)
    FULL = "full"    # Always use LLM for all documents


class SearchRequest(BaseModel):
    """Request model for search endpoint"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    max_results: Optional[int] = Field(default=20, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Similarity threshold")
    rerank_mode: Optional[RerankMode] = Field(default=RerankMode.SMART, description="Re-ranking mode (smart or full)")
    

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


class PipelineEfficiency(BaseModel):
    """Pipeline stage efficiency percentages"""
    extraction_pct: float = 0
    rewrite_pct: float = 0
    retrieval_pct: float = 0
    fusion_pct: float = 0
    rerank_pct: float = 0
    answer_pct: float = 0


class PerformanceMetrics(BaseModel):
    """Performance metrics for search pipeline"""
    total_time: float
    extraction_time: float
    rewrite_time: float
    retrieval_time: float
    fusion_time: float
    rerank_time: float = 0
    answer_time: float
    pipeline_efficiency: PipelineEfficiency
    rerank_mode: str = "smart"
    rerank_decision: str = ""
    tokens_used: int = 0


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
    components: Dict[str, Any]
    database: Dict[str, Any]
    embedding: Dict[str, Any]
    hybrid_enabled: bool
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)