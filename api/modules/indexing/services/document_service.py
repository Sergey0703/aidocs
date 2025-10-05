#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# api/modules/indexing/services/document_service.py
# Complete implementation with database integration

import logging
import sys
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

from ..models.schemas import (
    DocumentListItem,
    DocumentInfo,
    DocumentChunk,
)

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for managing document operations with full database integration"""
    
    def __init__(self):
        self._db_manager = None
        self._config = None
        
        # Add backend path to sys.path
        self._setup_backend_path()
        
        logger.info("✅ DocumentService initialized")
    
    def _setup_backend_path(self):
        """Add rag_indexer to Python path"""
        try:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent.parent
            backend_path = project_root / "rag_indexer"
            
            if backend_path.exists():
                sys.path.insert(0, str(backend_path))
                logger.info(f"Added backend path: {backend_path}")
            else:
                logger.warning(f"Backend path not found: {backend_path}")
        except Exception as e:
            logger.error(f"Failed to setup backend path: {e}")
    
    def _get_db_manager(self):
        """Lazy initialization of database manager"""
        if self._db_manager is None:
            try:
                from chunking_vectors.database_manager import create_database_manager
                from chunking_vectors.config import get_config
                
                self._config = get_config()
                
                # FIX: Override DOCUMENTS_DIR to use absolute path
                current_file = Path(__file__)
                project_root = current_file.parent.parent.parent.parent.parent
                markdown_dir = project_root / "rag_indexer" / "data" / "markdown"
                
                # Update config with correct path
                self._config.DOCUMENTS_DIR = str(markdown_dir)
                
                logger.info(f"📁 Using markdown directory: {markdown_dir}")
                
                # Create directory if it doesn't exist
                markdown_dir.mkdir(parents=True, exist_ok=True)
                
                self._db_manager = create_database_manager(
                    self._config.CONNECTION_STRING,
                    self._config.TABLE_NAME
                )
                
                logger.info("✅ Database manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database manager: {e}")
                raise
        
        return self._db_manager
    
    async def get_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "indexed_at",
        order: str = "desc"
    ) -> tuple[List[DocumentListItem], int, int, int]:
        """
        Get list of documents from database
        
        Returns:
            tuple: (documents, total_documents, total_chunks, total_characters)
        """
        try:
            db_manager = self._get_db_manager()
            
            import psycopg2
            conn = psycopg2.connect(self._config.CONNECTION_STRING)
            cur = conn.cursor()
            
            # Build ORDER BY clause
            valid_sort_columns = {
                'indexed_at': "MAX(metadata->>'indexed_at')",
                'file_name': "metadata->>'file_name'",
                'total_chunks': "COUNT(*)",
                'total_characters': "SUM(LENGTH(metadata->>'text'))"
            }
            
            sort_column = valid_sort_columns.get(sort_by, valid_sort_columns['indexed_at'])
            order_direction = 'DESC' if order.lower() == 'desc' else 'ASC'
            
            # Query database for documents
            query = f"""
                SELECT 
                    metadata->>'file_name' as filename,
                    COUNT(*) as total_chunks,
                    SUM(LENGTH(metadata->>'text')) as total_characters,
                    MAX(metadata->>'indexed_at') as indexed_at,
                    metadata->>'file_type' as file_type
                FROM vecs.{self._config.TABLE_NAME}
                WHERE metadata->>'file_name' IS NOT NULL
                GROUP BY metadata->>'file_name', metadata->>'file_type'
                ORDER BY {sort_column} {order_direction}
                LIMIT %s OFFSET %s
            """
            
            cur.execute(query, (limit, offset))
            rows = cur.fetchall()
            
            documents = []
            total_chunks = 0
            total_characters = 0
            
            for row in rows:
                filename, chunks, chars, indexed_at, file_type = row
                
                doc_item = DocumentListItem(
                    filename=filename,
                    total_chunks=chunks,
                    total_characters=chars or 0,
                    indexed_at=datetime.fromisoformat(indexed_at) if indexed_at else None,
                    file_type=file_type or "md",
                )
                
                documents.append(doc_item)
                total_chunks += chunks
                total_characters += chars or 0
            
            # Get total count
            cur.execute(f"""
                SELECT COUNT(DISTINCT metadata->>'file_name')
                FROM vecs.{self._config.TABLE_NAME}
                WHERE metadata->>'file_name' IS NOT NULL
            """)
            total_documents = cur.fetchone()[0]
            
            cur.close()
            conn.close()
            
            logger.info(f"Retrieved {len(documents)} documents (total: {total_documents})")
            
            return documents, total_documents, total_chunks, total_characters
            
        except Exception as e:
            logger.error(f"Failed to get documents: {e}", exc_info=True)
            return [], 0, 0, 0
    
    async def get_document_by_filename(
        self,
        filename: str,
        include_chunks: bool = False
    ) -> Optional[tuple[DocumentInfo, Optional[List[DocumentChunk]]]]:
        """
        Get document details by filename
        
        Returns:
            tuple: (document_info, chunks) or None if not found
        """
        try:
            db_manager = self._get_db_manager()
            
            import psycopg2
            conn = psycopg2.connect(self._config.CONNECTION_STRING)
            cur = conn.cursor()
            
            # Get document metadata
            cur.execute(f"""
                SELECT 
                    metadata->>'file_name' as filename,
                    metadata->>'file_path' as file_path,
                    metadata->>'file_type' as file_type,
                    COUNT(*) as total_chunks,
                    SUM(LENGTH(metadata->>'text')) as total_characters,
                    AVG(LENGTH(metadata->>'text')) as avg_chunk_length,
                    MAX(metadata->>'indexed_at') as indexed_at
                FROM vecs.{self._config.TABLE_NAME}
                WHERE metadata->>'file_name' = %s
                GROUP BY metadata->>'file_name', metadata->>'file_path', metadata->>'file_type'
            """, (filename,))
            
            row = cur.fetchone()
            
            if not row:
                cur.close()
                conn.close()
                logger.warning(f"Document not found: {filename}")
                return None
            
            fname, file_path, file_type, total_chunks, total_chars, avg_len, indexed_at = row
            
            # Get chunk indices
            cur.execute(f"""
                SELECT (metadata->>'chunk_index')::int
                FROM vecs.{self._config.TABLE_NAME}
                WHERE metadata->>'file_name' = %s
                  AND metadata->>'chunk_index' IS NOT NULL
                ORDER BY (metadata->>'chunk_index')::int
            """, (filename,))
            
            chunk_indices = [row[0] for row in cur.fetchall() if row[0] is not None]
            
            document = DocumentInfo(
                filename=fname,
                file_path=file_path,
                file_type=file_type or "md",
                total_chunks=total_chunks,
                chunk_indices=chunk_indices,
                total_characters=total_chars or 0,
                avg_chunk_length=avg_len or 0.0,
                indexed_at=datetime.fromisoformat(indexed_at) if indexed_at else None,
            )
            
            chunks = None
            if include_chunks:
                cur.execute(f"""
                    SELECT 
                        COALESCE((metadata->>'chunk_index')::int, 0) as chunk_index,
                        metadata->>'text' as content,
                        metadata
                    FROM vecs.{self._config.TABLE_NAME}
                    WHERE metadata->>'file_name' = %s
                    ORDER BY COALESCE((metadata->>'chunk_index')::int, 0)
                """, (filename,))
                
                chunks = []
                for row in cur.fetchall():
                    chunk_idx, content, metadata = row
                    
                    chunk = DocumentChunk(
                        chunk_index=chunk_idx if chunk_idx is not None else 0,
                        content=content or "",
                        content_length=len(content) if content else 0,
                        metadata=metadata or {},
                    )
                    chunks.append(chunk)
                
                logger.info(f"Retrieved {len(chunks)} chunks for {filename}")
            
            cur.close()
            conn.close()
            
            return document, chunks
            
        except Exception as e:
            logger.error(f"Failed to get document {filename}: {e}", exc_info=True)
            return None
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive document statistics
        
        Returns:
            dict: Statistics including totals, averages, distributions
        """
        try:
            db_manager = self._get_db_manager()
            
            import psycopg2
            conn = psycopg2.connect(self._config.CONNECTION_STRING)
            cur = conn.cursor()
            
            # Total documents and chunks
            cur.execute(f"""
                SELECT 
                    COUNT(DISTINCT metadata->>'file_name') as total_documents,
                    COUNT(*) as total_chunks,
                    SUM(LENGTH(metadata->>'text')) as total_characters
                FROM vecs.{self._config.TABLE_NAME}
                WHERE metadata->>'file_name' IS NOT NULL
            """)
            total_docs, total_chunks, total_chars = cur.fetchone()
            
            # Chunks per document stats
            cur.execute(f"""
                SELECT 
                    AVG(chunk_count) as avg_chunks,
                    MIN(chunk_count) as min_chunks,
                    MAX(chunk_count) as max_chunks
                FROM (
                    SELECT COUNT(*) as chunk_count
                    FROM vecs.{self._config.TABLE_NAME}
                    WHERE metadata->>'file_name' IS NOT NULL
                    GROUP BY metadata->>'file_name'
                ) as subquery
            """)
            avg_chunks, min_chunks, max_chunks = cur.fetchone()
            
            # File types distribution
            cur.execute(f"""
                SELECT 
                    COALESCE(metadata->>'file_type', 'unknown') as file_type,
                    COUNT(DISTINCT metadata->>'file_name') as count
                FROM vecs.{self._config.TABLE_NAME}
                WHERE metadata->>'file_name' IS NOT NULL
                GROUP BY metadata->>'file_type'
            """)
            file_types = {row[0]: row[1] for row in cur.fetchall()}
            
            # Size distribution
            cur.execute(f"""
                SELECT 
                    CASE 
                        WHEN total_chars < 1000 THEN 'small'
                        WHEN total_chars < 5000 THEN 'medium'
                        WHEN total_chars < 20000 THEN 'large'
                        ELSE 'very_large'
                    END as size_category,
                    COUNT(*) as count
                FROM (
                    SELECT SUM(LENGTH(metadata->>'text')) as total_chars
                    FROM vecs.{self._config.TABLE_NAME}
                    WHERE metadata->>'file_name' IS NOT NULL
                    GROUP BY metadata->>'file_name'
                ) as subquery
                GROUP BY size_category
            """)
            
            size_distribution = {"small": 0, "medium": 0, "large": 0, "very_large": 0}
            for row in cur.fetchall():
                size_distribution[row[0]] = row[1]
            
            cur.close()
            conn.close()
            
            stats = {
                "total_documents": total_docs or 0,
                "total_chunks": total_chunks or 0,
                "total_characters": total_chars or 0,
                "avg_chunks_per_document": float(avg_chunks) if avg_chunks else 0.0,
                "min_chunks": min_chunks or 0,
                "max_chunks": max_chunks or 0,
                "file_types": file_types,
                "size_distribution": size_distribution
            }
            
            logger.info(f"Retrieved stats: {total_docs} documents, {total_chunks} chunks")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get document stats: {e}", exc_info=True)
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_characters": 0,
                "avg_chunks_per_document": 0.0,
                "min_chunks": 0,
                "max_chunks": 0,
                "file_types": {},
                "size_distribution": {"small": 0, "medium": 0, "large": 0, "very_large": 0}
            }
    
    async def search_documents(
        self,
        filename_pattern: Optional[str] = None,
        min_chunks: Optional[int] = None,
        max_chunks: Optional[int] = None,
        indexed_after: Optional[datetime] = None,
        limit: int = 100
    ) -> tuple[List[DocumentListItem], int, int, int]:
        """
        Search documents by metadata
        
        Returns:
            tuple: (documents, total_documents, total_chunks, total_characters)
        """
        try:
            import psycopg2
            conn = psycopg2.connect(self._config.CONNECTION_STRING)
            cur = conn.cursor()
            
            # Build WHERE clause
            where_conditions = ["metadata->>'file_name' IS NOT NULL"]
            params = []
            
            if filename_pattern:
                where_conditions.append("metadata->>'file_name' ILIKE %s")
                params.append(f"%{filename_pattern}%")
            
            if indexed_after:
                where_conditions.append("metadata->>'indexed_at' > %s")
                params.append(indexed_after.isoformat())
            
            where_clause = " AND ".join(where_conditions)
            
            # Query with filters
            query = f"""
                SELECT 
                    metadata->>'file_name' as filename,
                    COUNT(*) as total_chunks,
                    SUM(LENGTH(metadata->>'text')) as total_characters,
                    MAX(metadata->>'indexed_at') as indexed_at,
                    metadata->>'file_type' as file_type
                FROM vecs.{self._config.TABLE_NAME}
                WHERE {where_clause}
                GROUP BY metadata->>'file_name', metadata->>'file_type'
            """
            
            # Add chunk filters in HAVING clause
            having_conditions = []
            if min_chunks is not None:
                having_conditions.append(f"COUNT(*) >= {min_chunks}")
            if max_chunks is not None:
                having_conditions.append(f"COUNT(*) <= {max_chunks}")
            
            if having_conditions:
                query += " HAVING " + " AND ".join(having_conditions)
            
            query += f" LIMIT {limit}"
            
            cur.execute(query, params)
            rows = cur.fetchall()
            
            documents = []
            total_chunks = 0
            total_characters = 0
            
            for row in rows:
                filename, chunks, chars, indexed_at, file_type = row
                
                doc_item = DocumentListItem(
                    filename=filename,
                    total_chunks=chunks,
                    total_characters=chars or 0,
                    indexed_at=datetime.fromisoformat(indexed_at) if indexed_at else None,
                    file_type=file_type or "md",
                )
                
                documents.append(doc_item)
                total_chunks += chunks
                total_characters += chars or 0
            
            cur.close()
            conn.close()
            
            logger.info(f"Search found {len(documents)} documents")
            
            return documents, len(documents), total_chunks, total_characters
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}", exc_info=True)
            return [], 0, 0, 0
    
    async def delete_document(
        self,
        filename: str,
        delete_chunks: bool = True
    ) -> tuple[bool, int]:
        """
        Delete document from database
        
        Returns:
            tuple: (success, chunks_deleted)
        """
        try:
            db_manager = self._get_db_manager()
            
            import psycopg2
            conn = psycopg2.connect(self._config.CONNECTION_STRING)
            cur = conn.cursor()
            
            # Check if document exists
            cur.execute(f"""
                SELECT COUNT(*) 
                FROM vecs.{self._config.TABLE_NAME}
                WHERE metadata->>'file_name' = %s
            """, (filename,))
            
            count = cur.fetchone()[0]
            
            if count == 0:
                cur.close()
                conn.close()
                logger.warning(f"Document not found: {filename}")
                return False, 0
            
            # Delete document
            chunks_deleted = 0
            if delete_chunks:
                cur.execute(f"""
                    DELETE FROM vecs.{self._config.TABLE_NAME}
                    WHERE metadata->>'file_name' = %s
                """, (filename,))
                
                chunks_deleted = cur.rowcount
                conn.commit()
                
                logger.info(f"Deleted document {filename}: {chunks_deleted} chunks removed")
            
            cur.close()
            conn.close()
            
            return True, chunks_deleted
            
        except Exception as e:
            logger.error(f"Failed to delete document {filename}: {e}", exc_info=True)
            return False, 0
    
    async def get_document_chunks(
        self,
        filename: str,
        limit: int = 100,
        offset: int = 0
    ) -> Optional[tuple[DocumentInfo, List[DocumentChunk]]]:
        """
        Get chunks for a specific document with pagination
        
        Returns:
            tuple: (document_info, chunks) or None if not found
        """
        try:
            # Get document info first
            result = await self.get_document_by_filename(filename, include_chunks=False)
            
            if result is None:
                return None
            
            document_info, _ = result
            
            # Get chunks with pagination
            import psycopg2
            conn = psycopg2.connect(self._config.CONNECTION_STRING)
            cur = conn.cursor()
            
            cur.execute(f"""
                SELECT 
                    COALESCE((metadata->>'chunk_index')::int, 0) as chunk_index,
                    metadata->>'text' as content,
                    metadata
                FROM vecs.{self._config.TABLE_NAME}
                WHERE metadata->>'file_name' = %s
                ORDER BY COALESCE((metadata->>'chunk_index')::int, 0)
                LIMIT %s OFFSET %s
            """, (filename, limit, offset))
            
            chunks = []
            for row in cur.fetchall():
                chunk_idx, content, metadata = row
                
                chunk = DocumentChunk(
                    chunk_index=chunk_idx if chunk_idx is not None else 0,
                    content=content or "",
                    content_length=len(content) if content else 0,
                    metadata=metadata or {},
                )
                chunks.append(chunk)
            
            cur.close()
            conn.close()
            
            logger.info(f"Retrieved {len(chunks)} chunks for {filename} (offset: {offset})")
            
            return document_info, chunks
            
        except Exception as e:
            logger.error(f"Failed to get chunks for {filename}: {e}", exc_info=True)
            return None
    
    async def upload_document(
        self,
        filename: str,
        content: bytes,
        auto_index: bool = True
    ) -> DocumentInfo:
        """
        Upload new document and optionally trigger indexing
        
        Returns:
            DocumentInfo: Uploaded document information
        """
        try:
            # Get correct path from config
            db_manager = self._get_db_manager()
            docs_dir = Path(self._config.DOCUMENTS_DIR)
            docs_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = docs_dir / filename
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(content)
            
            logger.info(f"Uploaded document: {filename} ({len(content)} bytes)")
            
            # Return document info
            document = DocumentInfo(
                filename=filename,
                file_path=str(file_path),
                file_type="md" if filename.endswith('.md') else "unknown",
                total_chunks=0,
                chunk_indices=[],
                total_characters=len(content),
                avg_chunk_length=0.0,
                indexed_at=None,  # Not indexed yet
            )
            
            # TODO: Optionally trigger indexing if auto_index=True
            if auto_index:
                logger.info(f"Auto-indexing requested for {filename}")
                # This could call IndexingService to queue the file
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to upload document {filename}: {e}", exc_info=True)
            raise
    
    async def get_missing_documents(self) -> tuple[List[str], int, int, int, float]:
        """
        Get files present in directory but missing from database
        
        Returns:
            tuple: (missing_files, total_missing, total_in_directory, 
                   total_in_database, success_rate)
        """
        try:
            db_manager = self._get_db_manager()
            
            # Use database_manager's analyze_directory_vs_database method
            analysis_results = db_manager.analyze_directory_vs_database(
                self._config.DOCUMENTS_DIR,
                recursive=True,
                blacklist_directories=self._config.BLACKLIST_DIRECTORIES
            )
            
            missing_files = analysis_results['missing_files_detailed']
            total_missing = analysis_results['files_missing_from_db']
            total_in_directory = analysis_results['total_files_in_directory']
            total_in_database = analysis_results['files_successfully_in_db']
            success_rate = analysis_results['success_rate']
            
            logger.info(f"Missing files analysis: {total_missing} missing out of {total_in_directory}")
            
            return (
                missing_files,
                total_missing,
                total_in_directory,
                total_in_database,
                success_rate
            )
            
        except Exception as e:
            logger.error(f"Failed to get missing documents: {e}", exc_info=True)
            return [], 0, 0, 0, 0.0
    
    async def check_document_exists(self, filename: str) -> bool:
        """
        Check if document exists in database
        
        Returns:
            bool: True if document exists
        """
        try:
            import psycopg2
            conn = psycopg2.connect(self._config.CONNECTION_STRING)
            cur = conn.cursor()
            
            cur.execute(f"""
                SELECT COUNT(*) 
                FROM vecs.{self._config.TABLE_NAME}
                WHERE metadata->>'file_name' = %s
            """, (filename,))
            
            count = cur.fetchone()[0]
            
            cur.close()
            conn.close()
            
            return count > 0
            
        except Exception as e:
            logger.error(f"Failed to check if document exists {filename}: {e}", exc_info=True)
            return False
    
    async def get_document_count(self) -> int:
        """
        Get total number of documents in database
        
        Returns:
            int: Total document count
        """
        try:
            import psycopg2
            conn = psycopg2.connect(self._config.CONNECTION_STRING)
            cur = conn.cursor()
            
            cur.execute(f"""
                SELECT COUNT(DISTINCT metadata->>'file_name')
                FROM vecs.{self._config.TABLE_NAME}
                WHERE metadata->>'file_name' IS NOT NULL
            """)
            
            count = cur.fetchone()[0]
            
            cur.close()
            conn.close()
            
            return count or 0
            
        except Exception as e:
            logger.error(f"Failed to get document count: {e}", exc_info=True)
            return 0


# Singleton instance
_document_service: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """Get or create document service singleton"""
    global _document_service
    
    if _document_service is None:
        _document_service = DocumentService()
    
    return _document_service