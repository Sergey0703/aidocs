#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registry Manager - Bridge between document registry and vector indexing
Manages document_registry entries for the indexing pipeline
"""

import logging
import psycopg2
import psycopg2.extras
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class RegistryManager:
    """Manages document registry entries for indexing"""
    
    def __init__(self, connection_string: str):
        """
        Initialize registry manager
        
        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string
    
    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.connection_string)
    
    def get_or_create_registry_entry(
        self,
        file_path: str,
        document_type: Optional[str] = None,
        vehicle_id: Optional[str] = None,
        extracted_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Get existing or create new registry entry for a document
        
        Args:
            file_path: Path to the document file
            document_type: Type of document (e.g., 'insurance', 'nct', etc.)
            vehicle_id: UUID of associated vehicle (optional)
            extracted_data: Extracted metadata (optional)
        
        Returns:
            str: registry_id (UUID) or None if failed
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Try to find existing entry
            cur.execute("""
                SELECT id FROM vecs.document_registry
                WHERE file_path = %s
            """, (file_path,))
            
            result = cur.fetchone()
            
            if result:
                registry_id = str(result['id'])
                logger.debug(f"Found existing registry entry: {registry_id} for {Path(file_path).name}")
            else:
                # Create new entry
                cur.execute("""
                    INSERT INTO vecs.document_registry 
                    (file_path, document_type, vehicle_id, status, extracted_data)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    file_path,
                    document_type or 'unknown',
                    vehicle_id,
                    'indexed',  # Status: indexed (as we're about to index it)
                    psycopg2.extras.Json(extracted_data) if extracted_data else None
                ))
                
                result = cur.fetchone()
                registry_id = str(result['id'])
                
                conn.commit()
                logger.info(f"Created new registry entry: {registry_id} for {Path(file_path).name}")
            
            cur.close()
            conn.close()
            
            return registry_id
            
        except Exception as e:
            logger.error(f"Failed to get/create registry entry for {file_path}: {e}", exc_info=True)
            return None
    
    def update_registry_status(self, registry_id: str, status: str):
        """
        Update status of a registry entry
        
        Args:
            registry_id: UUID of registry entry
            status: New status (e.g., 'indexed', 'failed', etc.)
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                UPDATE vecs.document_registry
                SET status = %s, updated_at = NOW()
                WHERE id = %s
            """, (status, registry_id))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.debug(f"Updated registry {registry_id} status to: {status}")
            
        except Exception as e:
            logger.error(f"Failed to update registry status: {e}")
    
    def cleanup_orphaned_chunks(self, registry_id: str):
        """
        Clean up existing chunks for a registry entry before re-indexing
        
        Args:
            registry_id: UUID of registry entry
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                DELETE FROM vecs.documents
                WHERE registry_id = %s
            """, (registry_id,))
            
            deleted_count = cur.rowcount
            conn.commit()
            cur.close()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} existing chunks for registry {registry_id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup chunks for registry {registry_id}: {e}")


def create_registry_manager(connection_string: str) -> RegistryManager:
    """
    Factory function to create a RegistryManager instance
    
    Args:
        connection_string: PostgreSQL connection string
    
    Returns:
        RegistryManager: Initialized manager
    """
    return RegistryManager(connection_string)