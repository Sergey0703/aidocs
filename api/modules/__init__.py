# api/modules/__init__.py
# API modules registry

from . import search

# Registry of all available modules
AVAILABLE_MODULES = {
    "search": {
        "name": "Search",
        "version": "1.0.0",
        "status": "active",
        "description": "Hybrid search with AI re-ranking",
        "routers": [search.search_router, search.system_router]
    },
    # Future modules will be added here:
    # "indexing": {...},
    # "documents": {...},
    # "templates": {...},
    # "verification": {...},
}


def get_module_info(module_name: str):
    """Get information about a specific module"""
    return AVAILABLE_MODULES.get(module_name)


def get_active_modules():
    """Get list of all active modules"""
    return {
        name: info 
        for name, info in AVAILABLE_MODULES.items() 
        if info["status"] == "active"
    }


__all__ = [
    "search",
    "AVAILABLE_MODULES",
    "get_module_info",
    "get_active_modules",
]