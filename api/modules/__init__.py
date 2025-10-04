# api/modules/__init__.py
# API modules registry

from . import search
from . import indexing

# Registry of all available modules
AVAILABLE_MODULES = {
    "search": {
        "name": "Search",
        "version": "1.0.0",
        "status": "active",
        "description": "Hybrid search with AI re-ranking",
        "routers": [search.search_router, search.system_router]
    },
    "indexing": {
        "name": "Indexing",
        "version": "1.0.0",
        "status": "active",
        "description": "Document indexing and conversion",
        "routers": [
            indexing.indexing_router,
            indexing.documents_router,
            indexing.conversion_router,
            indexing.monitoring_router
        ]
    },
    # Future modules:
    # "templates": {...},
    # "verification": {...},
}