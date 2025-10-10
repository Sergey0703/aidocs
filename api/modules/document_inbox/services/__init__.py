# api/modules/document_inbox/services/__init__.py
# Document Inbox services initialization

from .vrn_extraction_service import (
    VRNExtractionService,
    VRNExtractionResult,
    get_vrn_extraction_service
)

__all__ = [
    'VRNExtractionService',
    'VRNExtractionResult',
    'get_vrn_extraction_service'
]