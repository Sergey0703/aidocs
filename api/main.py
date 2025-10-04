# api/main.py
# Main FastAPI application entry point
# Unified API for Document Intelligence Platform

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add streamlit-rag to Python path for backend imports
backend_path = Path(__file__).parent.parent / "streamlit-rag"
sys.path.insert(0, str(backend_path))

# Import from modules
from api.modules import search, AVAILABLE_MODULES
from api.core.dependencies import initialize_system_components

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting Document Intelligence Platform API...")
    logger.info("📁 Backend path: %s", backend_path)
    
    try:
        # Initialize system components
        initialize_system_components()
        logger.info("✅ System components initialized successfully")
    except Exception as e:
        logger.error("❌ Failed to initialize system: %s", e)
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Document Intelligence Platform API...")


# Create FastAPI application
app = FastAPI(
    title="Document Intelligence Platform",
    description="Unified API for document search, indexing, templates, and verification",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers from search module with prefix
app.include_router(
    search.search_router,
    prefix="/api/search",  # FIXED: добавлен prefix
    tags=["Search"]
)

app.include_router(
    search.system_router,
    prefix="/api/system",  # FIXED: добавлен prefix
    tags=["System"]
)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information
    """
    return {
        "name": "Document Intelligence Platform",
        "version": "1.0.0",
        "status": "operational",
        "description": "Unified API for document search, indexing, templates, and verification",
        "modules": AVAILABLE_MODULES,
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    }


# Health check endpoint (alias for /api/system/health)
@app.get("/health", tags=["Root"])
async def health_check():
    """
    Quick health check endpoint
    """
    return {
        "status": "healthy",
        "service": "Document Intelligence Platform",
        "version": "1.0.0"
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for uncaught exceptions
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "message": "An internal server error occurred"
        }
    )


# Startup message
@app.on_event("startup")
async def startup_message():
    """
    Print startup message with useful information
    """
    logger.info("=" * 70)
    logger.info("📡 Document Intelligence Platform API")
    logger.info("=" * 70)
    logger.info("🔗 API Documentation: http://localhost:8000/docs")
    logger.info("📚 ReDoc: http://localhost:8000/redoc")
    logger.info("🔍 Search endpoint: http://localhost:8000/api/search")
    logger.info("❤️  Health check: http://localhost:8000/health")
    logger.info("=" * 70)


if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )