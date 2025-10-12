#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# api/modules/document_inbox/services/vrn_extraction_service.py
# VRN Extraction Service - extracts Vehicle Registration Numbers from documents

import logging
import re
import sys
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class VRNExtractionService:
    """Service for extracting VRN from document text using regex and AI"""
    
    def __init__(self):
        self._config = None
        self._openai_client = None
        logger.info("✅ VRNExtractionService initialized")
    
    def _setup_backend_path(self):
        """Add rag_indexer to Python path"""
        try:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent.parent
            backend_path = project_root / "rag_indexer"
            
            if backend_path.exists() and str(backend_path) not in sys.path:
                sys.path.insert(0, str(backend_path))
                logger.debug(f"Added backend path: {backend_path}")
        except Exception as e:
            logger.error(f"Failed to setup backend path: {e}")
    
    def _get_config(self):
        """Lazy initialization of configuration"""
        if self._config is None:
            self._setup_backend_path()
            from chunking_vectors.config import get_config
            self._config = get_config()
        return self._config
    
    def _get_openai_client(self):
        """Lazy initialization of OpenAI client"""
        if self._openai_client is None:
            try:
                from openai import OpenAI
                config = self._get_config()
                self._openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
                logger.debug("✅ OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self._openai_client = None
        return self._openai_client
    
    # ========================================================================
    # IRISH VRN REGEX PATTERNS
    # ========================================================================
    
    @staticmethod
    def _get_vrn_patterns() -> List[re.Pattern]:
        """
        Get compiled regex patterns for Irish VRN formats
        
        Irish VRN formats:
        - Modern (2013+): YY-C-NNNNN (e.g., 191-D-12345, 24-KY-999)
        - Legacy: YY-C-NNNN or C-NNNNN (e.g., 06-D-1234, D-12345)
        """
        return [
            # Modern format: YY-C-NNNNN (2013+)
            # Year (2 digits) - County (1-2 letters) - Number (1-6 digits)
            re.compile(r'\b(\d{2,3})-([A-Z]{1,2})-(\d{1,6})\b', re.IGNORECASE),
            
            # Legacy format: YY-C-NNNN
            re.compile(r'\b(\d{2})-([A-Z]{1,2})-(\d{1,5})\b', re.IGNORECASE),
            
            # Legacy format: C-NNNNN
            re.compile(r'\b([A-Z]{1,2})-(\d{1,6})\b', re.IGNORECASE),
            
            # Format without dashes: YYCNNNNN
            re.compile(r'\b(\d{2,3})([A-Z]{1,2})(\d{1,6})\b', re.IGNORECASE),
        ]
    
    @staticmethod
    def _normalize_vrn(vrn: str) -> str:
        """
        Normalize VRN to standard format: YY-C-NNNNN or C-NNNNN
        
        Examples:
            191D12345 → 191-D-12345
            06D1234 → 06-D-1234
            D12345 → D-12345
        """
        # Remove all spaces and convert to uppercase
        vrn = vrn.upper().strip().replace(' ', '')
        
        # If already has dashes in correct format, return as is
        if re.match(r'^\d{2,3}-[A-Z]{1,2}-\d{1,6}$', vrn):
            return vrn
        
        if re.match(r'^[A-Z]{1,2}-\d{1,6}$', vrn):
            return vrn
        
        # Try to parse and add dashes
        # Modern format: 191D12345 → 191-D-12345
        match = re.match(r'^(\d{2,3})([A-Z]{1,2})(\d{1,6})$', vrn)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        
        # Legacy format: D12345 → D-12345
        match = re.match(r'^([A-Z]{1,2})(\d{1,6})$', vrn)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        
        # Return as is if can't parse
        return vrn
    
    def extract_vrn_from_text(self, text: str) -> Optional[str]:
        """
        Extract VRN from text using regex patterns
        
        Args:
            text: Document text to search
        
        Returns:
            Normalized VRN string or None if not found
        """
        if not text:
            return None
        
        patterns = self._get_vrn_patterns()
        
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                # Get first match
                match = matches[0]
                
                # Reconstruct VRN from match groups
                if isinstance(match, tuple):
                    vrn_raw = ''.join(match)
                else:
                    vrn_raw = match
                
                # Normalize to standard format
                vrn = self._normalize_vrn(vrn_raw)
                
                logger.debug(f"✅ VRN found via regex: {vrn}")
                return vrn
        
        logger.debug("❌ No VRN found via regex")
        return None
    
    def extract_vrn_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract VRN from filename
        
        Examples:
            191-D-12345_insurance.pdf → 191-D-12345
            06-D-1234_nct.pdf → 06-D-1234
        """
        if not filename:
            return None
        
        patterns = self._get_vrn_patterns()
        
        for pattern in patterns:
            match = pattern.search(filename)
            if match:
                vrn_raw = match.group(0)
                vrn = self._normalize_vrn(vrn_raw)
                logger.debug(f"✅ VRN found in filename: {vrn}")
                return vrn
        
        return None
    
    async def extract_vrn_with_ai(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract VRN, make, and model using OpenAI
        
        Args:
            text: Document text to analyze
        
        Returns:
            {
                'vrn': '191-D-12345',
                'make': 'Toyota',
                'model': 'Corolla'
            }
            or None if extraction failed
        """
        try:
            client = self._get_openai_client()
            if not client:
                logger.warning("OpenAI client not available")
                return None
            
            # Limit text to first 2000 characters for efficiency
            text_snippet = text[:2000] if len(text) > 2000 else text
            
            prompt = f"""Extract vehicle information from this Irish document.

Document text:
{text_snippet}

Extract:
1. VRN (Vehicle Registration Number) - Irish format like 191-D-12345, 06-D-1234, or D-12345
2. Make (vehicle manufacturer)
3. Model (vehicle model)

Respond ONLY with JSON in this exact format:
{{"vrn": "191-D-12345", "make": "Toyota", "model": "Corolla"}}

If any field is not found, use null. If no vehicle information found, respond with: {{"vrn": null, "make": null, "model": null}}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. Extract vehicle information from Irish documents. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=150
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            result = json.loads(result_text)
            
            # Normalize VRN if found
            if result.get('vrn'):
                result['vrn'] = self._normalize_vrn(result['vrn'])
                logger.info(f"✅ AI extracted: VRN={result['vrn']}, Make={result.get('make')}, Model={result.get('model')}")
                return result
            else:
                logger.debug("❌ AI found no VRN")
                return None
            
        except Exception as e:
            logger.error(f"AI extraction failed: {e}", exc_info=True)
            return None
    
    # ========================================================================
    # DOCUMENT TEXT RETRIEVAL
    # ========================================================================
    
    async def _get_document_text(self, filename: str) -> Optional[str]:
        """
        Get document text from vecs.documents table
        
        Args:
            filename: Document filename to search for
        
        Returns:
            Combined text from all chunks or None
        """
        try:
            import psycopg2
            import psycopg2.extras
            
            config = self._get_config()
            conn = psycopg2.connect(config.CONNECTION_STRING)
            
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        metadata->>'text' as text,
                        metadata->>'chunk_index' as chunk_index
                    FROM vecs.documents
                    WHERE metadata->>'file_name' = %s
                    ORDER BY (metadata->>'chunk_index')::int
                """, (filename,))
                
                chunks = cur.fetchall()
            
            conn.close()
            
            if not chunks:
                logger.warning(f"No chunks found for document: {filename}")
                return None
            
            # Combine all chunk texts
            full_text = ' '.join([chunk['text'] for chunk in chunks if chunk['text']])
            
            logger.debug(f"📄 Retrieved {len(chunks)} chunks, total length: {len(full_text)} chars")
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to get document text for {filename}: {e}", exc_info=True)
            return None
    
    # ========================================================================
    # REGISTRY UPDATE
    # ========================================================================
    
    async def _update_registry_with_vrn(
        self,
        registry_id: str,
        vrn: Optional[str],
        make: Optional[str] = None,
        model: Optional[str] = None,
        extraction_method: str = 'none'
    ) -> bool:
        """
        Update document_registry with extracted VRN data and status
        
        Args:
            registry_id: Document registry UUID
            vrn: Extracted VRN (or None if not found)
            make: Vehicle make (optional)
            model: Vehicle model (optional)
            extraction_method: 'regex', 'ai', 'filename', or 'none'
        
        Returns:
            bool: Success status
        """
        try:
            import psycopg2
            import psycopg2.extras
            
            config = self._get_config()
            conn = psycopg2.connect(config.CONNECTION_STRING)
            
            # Determine new status based on VRN presence
            if vrn:
                new_status = 'predassigned'  # VRN found - ready for auto-linking
                extracted_data = {
                    'vrn': vrn,
                    'extraction_method': extraction_method
                }
                if make:
                    extracted_data['make'] = make
                if model:
                    extracted_data['model'] = model
                
                logger.info(f"✅ Setting status='predassigned' for registry {registry_id} with VRN={vrn}")
            else:
                new_status = 'unassigned'  # No VRN - needs manual assignment
                extracted_data = {
                    'extraction_method': extraction_method
                }
                logger.info(f"⚠️ Setting status='unassigned' for registry {registry_id} (no VRN found)")
            
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE vecs.document_registry
                    SET 
                        extracted_data = extracted_data || %s::jsonb,
                        status = %s
                    WHERE id = %s
                """, (
                    psycopg2.extras.Json(extracted_data),
                    new_status,
                    registry_id
                ))
                
                affected = cur.rowcount
                conn.commit()
            
            conn.close()
            
            if affected > 0:
                logger.debug(f"✅ Updated registry {registry_id}: status={new_status}, method={extraction_method}")
                return True
            else:
                logger.warning(f"Registry {registry_id} not found")
                return False
            
        except Exception as e:
            logger.error(f"Failed to update registry {registry_id}: {e}", exc_info=True)
            return False
    
    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================
    
    async def process_document(
        self,
        registry_id: str,
        filename: str,
        use_ai: bool = True
    ) -> Tuple[bool, Optional[str], str]:
        """
        Process a single document to extract VRN
        
        Args:
            registry_id: Document registry UUID
            filename: Document filename
            use_ai: Whether to use AI if regex fails
        
        Returns:
            Tuple of (success, vrn, extraction_method)
        """
        try:
            logger.debug(f"🔍 Processing document: {filename}")
            
            # Try 1: Extract from filename
            vrn = self.extract_vrn_from_filename(filename)
            if vrn:
                await self._update_registry_with_vrn(registry_id, vrn, extraction_method='filename')
                return (True, vrn, 'filename')
            
            # Try 2: Get document text and extract with regex
            text = await self._get_document_text(filename)
            if text:
                vrn = self.extract_vrn_from_text(text)
                if vrn:
                    await self._update_registry_with_vrn(registry_id, vrn, extraction_method='regex')
                    return (True, vrn, 'regex')
                
                # Try 3: Use AI if enabled and regex failed
                if use_ai:
                    ai_result = await self.extract_vrn_with_ai(text)
                    if ai_result and ai_result.get('vrn'):
                        await self._update_registry_with_vrn(
                            registry_id,
                            ai_result['vrn'],
                            make=ai_result.get('make'),
                            model=ai_result.get('model'),
                            extraction_method='ai'
                        )
                        return (True, ai_result['vrn'], 'ai')
            
            # No VRN found - update status to 'unassigned'
            await self._update_registry_with_vrn(registry_id, None, extraction_method='none')
            return (True, None, 'none')
            
        except Exception as e:
            logger.error(f"Failed to process document {filename}: {e}", exc_info=True)
            return (False, None, 'error')
    
    async def process_batch(
        self,
        document_ids: Optional[List[str]] = None,
        use_ai: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple documents to extract VRN
        
        Args:
            document_ids: List of document registry IDs to process (None = all with status='processed')
            use_ai: Whether to use AI if regex fails
        
        Returns:
            Statistics dictionary
        """
        try:
            import psycopg2
            import psycopg2.extras
            
            config = self._get_config()
            conn = psycopg2.connect(config.CONNECTION_STRING)
            
            # Get documents to process
            if document_ids:
                # Process specific documents
                # Convert string UUIDs to proper format
                uuid_list = [str(doc_id) for doc_id in document_ids]
                
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Use explicit UUID casting
                    placeholders = ','.join(['%s::uuid' for _ in uuid_list])
                    query = f"""
                        SELECT id, raw_file_path
                        FROM vecs.document_registry
                        WHERE id IN ({placeholders})
                    """
                    cur.execute(query, uuid_list)
                    documents = cur.fetchall()
            else:
                # Process all documents with status='processed'
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT id, raw_file_path
                        FROM vecs.document_registry
                        WHERE status = 'processed'
                        ORDER BY uploaded_at DESC
                    """)
                    documents = cur.fetchall()
            
            conn.close()
            
            logger.info(f"📋 Found {len(documents)} documents to process for VRN extraction")
            
            # Statistics
            stats = {
                'total_processed': 0,
                'vrn_found': 0,
                'vrn_not_found': 0,
                'failed': 0,
                'extraction_methods': {
                    'regex': 0,
                    'ai': 0,
                    'filename': 0,
                    'none': 0
                }
            }
            
            # Process each document
            for doc in documents:
                success, vrn, method = await self.process_document(
                    str(doc['id']),
                    doc['raw_file_path'],
                    use_ai=use_ai
                )
                
                stats['total_processed'] += 1
                
                if success:
                    if vrn:
                        stats['vrn_found'] += 1
                        stats['extraction_methods'][method] += 1
                        logger.info(f"  ✅ {doc['raw_file_path']}: VRN={vrn} (method={method})")
                    else:
                        stats['vrn_not_found'] += 1
                        stats['extraction_methods']['none'] += 1
                        logger.info(f"  ⚠️ {doc['raw_file_path']}: No VRN found")
                else:
                    stats['failed'] += 1
                    logger.error(f"  ❌ {doc['raw_file_path']}: Processing failed")
            
            logger.info(
                f"📊 VRN Extraction Complete: "
                f"{stats['vrn_found']} found, "
                f"{stats['vrn_not_found']} not found, "
                f"{stats['failed']} failed"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            return {
                'total_processed': 0,
                'vrn_found': 0,
                'vrn_not_found': 0,
                'failed': 0,
                'extraction_methods': {
                    'regex': 0,
                    'ai': 0,
                    'filename': 0,
                    'none': 0
                }
            }


# Singleton
_vrn_extraction_service: Optional[VRNExtractionService] = None

def get_vrn_extraction_service() -> VRNExtractionService:
    """Get or create VRN extraction service singleton"""
    global _vrn_extraction_service
    
    if _vrn_extraction_service is None:
        _vrn_extraction_service = VRNExtractionService()
    
    return _vrn_extraction_service