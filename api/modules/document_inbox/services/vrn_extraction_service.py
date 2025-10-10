#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# api/modules/document_inbox/services/vrn_extraction_service.py
# VRN Extraction Service using Supabase chunks + AI

import logging
import asyncio
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import re

from config.settings import config
from api.modules.document_inbox.utils.vrn_patterns import VRNPatterns

logger = logging.getLogger(__name__)


@dataclass
class VRNExtractionResult:
    """Result of VRN extraction for a single document"""
    document_id: str
    filename: str
    vrn: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    document_type: Optional[str] = None
    extraction_method: str = "none"  # regex, ai, filename, none
    confidence: float = 0.0
    error: Optional[str] = None
    success: bool = False


class VRNExtractionService:
    """
    Service for extracting VRN (Vehicle Registration Number) from indexed documents.
    
    Uses:
    1. PostgreSQL direct access to vecs.documents for chunks
    2. Regex patterns for Irish VRN detection
    3. Google Gemini AI for fallback extraction
    4. Filename parsing as last resort
    """
    
    def __init__(self):
        self.config = config
        self.vrn_patterns = VRNPatterns()
        self.llm = None
        self._initialize_llm()
    
    
    def _initialize_llm(self):
        """Initialize Google Gemini LLM for AI extraction"""
        try:
            from llama_index.llms.google_genai import GoogleGenAI
            
            self.llm = GoogleGenAI(
                model=self.config.llm.main_model,
                api_key=self.config.llm.api_key,
                temperature=0.0,  # Deterministic for extraction
            )
            logger.info("✅ LLM initialized for VRN extraction")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            self.llm = None
    
    
    def _get_db_connection(self):
        """Get PostgreSQL database connection"""
        try:
            conn = psycopg2.connect(self.config.database.connection_string)
            return conn
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    
    async def get_chunks_from_db(self, filename: str) -> List[str]:
        """
        Get all text chunks for a document from vecs.documents table.
        
        Args:
            filename: Document filename (from document_registry.raw_file_path)
            
        Returns:
            List of text chunks sorted by chunk_index
        """
        try:
            conn = self._get_db_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            sql = """
            SELECT 
                metadata->>'text' as text,
                metadata->>'chunk_index' as chunk_index
            FROM vecs.documents
            WHERE metadata->>'file_name' = %s
            ORDER BY (metadata->>'chunk_index')::int
            """
            
            cur.execute(sql, (filename,))
            results = cur.fetchall()
            
            cur.close()
            conn.close()
            
            if not results:
                logger.warning(f"⚠️ No chunks found for: {filename}")
                return []
            
            chunks = [row['text'] for row in results if row['text']]
            logger.debug(f"📄 Retrieved {len(chunks)} chunks for {filename}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error retrieving chunks for {filename}: {e}")
            return []
    
    
    def combine_chunks(self, chunks: List[str], max_length: int = 50000) -> str:
        """
        Combine chunks into full text with length limit.
        
        Args:
            chunks: List of text chunks
            max_length: Maximum combined text length
            
        Returns:
            Combined text string
        """
        if not chunks:
            return ""
        
        full_text = " ".join(chunks)
        
        # Truncate if too long (for AI processing)
        if len(full_text) > max_length:
            logger.warning(f"⚠️ Text truncated from {len(full_text)} to {max_length} chars")
            full_text = full_text[:max_length]
        
        return full_text
    
    
    def extract_vrn_with_regex(self, text: str) -> Optional[str]:
        """
        Extract VRN using regex patterns.
        
        Args:
            text: Full document text
            
        Returns:
            VRN if found, None otherwise
        """
        if not text:
            return None
        
        vrn = self.vrn_patterns.extract_vrn(text)
        
        if vrn:
            logger.info(f"✅ Regex extracted VRN: {vrn}")
            return self.vrn_patterns.normalize_vrn(vrn)
        
        return None
    
    
    def extract_vrn_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract VRN from filename as fallback.
        
        Args:
            filename: Document filename
            
        Returns:
            VRN if found in filename, None otherwise
        """
        if not filename:
            return None
        
        # Try to extract VRN-like pattern from filename
        vrn = self.vrn_patterns.extract_vrn(filename)
        
        if vrn:
            logger.info(f"✅ Filename extracted VRN: {vrn} from {filename}")
            return self.vrn_patterns.normalize_vrn(vrn)
        
        return None
    
    
    async def extract_with_ai(self, text: str, filename: str = "") -> Dict[str, Optional[str]]:
        """
        Extract VRN, make, model, and document type using AI.
        
        Args:
            text: Document text
            filename: Document filename (for context)
            
        Returns:
            Dict with vrn, make, model, document_type
        """
        if not self.llm:
            logger.warning("⚠️ LLM not available for AI extraction")
            return {
                'vrn': None,
                'make': None,
                'model': None,
                'document_type': None
            }
        
        try:
            # Truncate text for AI processing (keep first 3000 chars)
            text_sample = text[:3000] if len(text) > 3000 else text
            
            prompt = f"""Extract vehicle information from this document text.

Document filename: {filename}
Document text:
{text_sample}

Extract the following information if present:
1. Vehicle Registration Number (VRN) - Irish format like 191-D-12345 or 06-D-12345
2. Vehicle Make (e.g., Toyota, Ford, BMW)
3. Vehicle Model (e.g., Corolla, Focus, 3 Series)
4. Document Type (e.g., insurance, motor_tax, nct, service, other)

Return ONLY valid JSON in this exact format:
{{
    "vrn": "191-D-12345" or null,
    "make": "Toyota" or null,
    "model": "Corolla" or null,
    "document_type": "insurance" or null
}}

If you cannot find any information, use null for that field.
Return ONLY the JSON, no other text."""

            logger.info("🤖 Calling AI for VRN extraction...")
            response = await self.llm.acomplete(prompt)
            response_text = response.text.strip()
            
            logger.debug(f"AI response: {response_text[:200]}...")
            
            # Parse JSON response
            result = self._parse_ai_response(response_text)
            
            if result.get('vrn'):
                logger.info(f"✅ AI extracted VRN: {result['vrn']}")
            else:
                logger.info("⚠️ AI did not find VRN")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ AI extraction failed: {e}")
            return {
                'vrn': None,
                'make': None,
                'model': None,
                'document_type': None
            }
    
    
    def _parse_ai_response(self, response_text: str) -> Dict[str, Optional[str]]:
        """
        Parse AI response JSON.
        
        Args:
            response_text: AI response text
            
        Returns:
            Parsed dict with vrn, make, model, document_type
        """
        import json
        
        try:
            # Try to extract JSON from response
            # Look for JSON object in response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                # Validate and normalize VRN if present
                if result.get('vrn'):
                    result['vrn'] = self.vrn_patterns.normalize_vrn(result['vrn'])
                
                return {
                    'vrn': result.get('vrn'),
                    'make': result.get('make'),
                    'model': result.get('model'),
                    'document_type': result.get('document_type')
                }
            else:
                logger.warning("⚠️ No JSON found in AI response")
                return {
                    'vrn': None,
                    'make': None,
                    'model': None,
                    'document_type': None
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse AI JSON: {e}")
            return {
                'vrn': None,
                'make': None,
                'model': None,
                'document_type': None
            }
    
    
    def detect_document_type(self, text: str, filename: str = "") -> Optional[str]:
        """
        Detect document type from text/filename.
        
        Args:
            text: Document text
            filename: Document filename
            
        Returns:
            Document type string or None
        """
        text_lower = text.lower()
        filename_lower = filename.lower()
        combined = f"{text_lower} {filename_lower}"
        
        # Document type keywords
        type_keywords = {
            'insurance': ['insurance', 'policy', 'cover', 'insured'],
            'motor_tax': ['motor tax', 'road tax', 'vehicle tax', 'disc'],
            'nct': ['nct', 'national car test', 'vehicle test'],
            'service': ['service', 'maintenance', 'repair', 'inspection'],
            'registration': ['registration', 'v5', 'logbook'],
            'other': []
        }
        
        for doc_type, keywords in type_keywords.items():
            if any(keyword in combined for keyword in keywords):
                logger.debug(f"📋 Detected document type: {doc_type}")
                return doc_type
        
        return 'other'
    
    
    async def process_document(
        self, 
        document_id: str, 
        filename: str,
        use_ai: bool = True
    ) -> VRNExtractionResult:
        """
        Process a single document to extract VRN and vehicle info.
        
        Process:
        1. Get chunks from database
        2. Try regex extraction
        3. Try filename extraction
        4. Try AI extraction (if enabled and needed)
        5. Return result
        
        Args:
            document_id: Document registry UUID
            filename: Document filename (raw_file_path)
            use_ai: Whether to use AI if regex fails
            
        Returns:
            VRNExtractionResult
        """
        logger.info(f"🔍 Processing document: {filename}")
        
        result = VRNExtractionResult(
            document_id=document_id,
            filename=filename
        )
        
        try:
            # Step 1: Get chunks from database
            chunks = await self.get_chunks_from_db(filename)
            
            if not chunks:
                result.error = "No chunks found in database"
                logger.warning(f"⚠️ {result.error} for {filename}")
                return result
            
            # Step 2: Combine chunks
            full_text = self.combine_chunks(chunks)
            
            if not full_text:
                result.error = "Empty text content"
                logger.warning(f"⚠️ {result.error} for {filename}")
                return result
            
            logger.debug(f"📄 Combined text length: {len(full_text)} chars")
            
            # Step 3: Try regex extraction first (fast and accurate)
            vrn = self.extract_vrn_with_regex(full_text)
            
            if vrn:
                result.vrn = vrn
                result.extraction_method = "regex"
                result.confidence = 0.95
                result.success = True
                
                # Detect document type
                result.document_type = self.detect_document_type(full_text, filename)
                
                logger.info(f"✅ Regex extraction successful: {vrn}")
                return result
            
            # Step 4: Try filename extraction
            vrn = self.extract_vrn_from_filename(filename)
            
            if vrn:
                result.vrn = vrn
                result.extraction_method = "filename"
                result.confidence = 0.80
                result.success = True
                
                # Detect document type
                result.document_type = self.detect_document_type(full_text, filename)
                
                logger.info(f"✅ Filename extraction successful: {vrn}")
                return result
            
            # Step 5: Try AI extraction (if enabled)
            if use_ai and self.llm:
                logger.info("🤖 Trying AI extraction...")
                
                ai_result = await self.extract_with_ai(full_text, filename)
                
                if ai_result.get('vrn'):
                    result.vrn = ai_result['vrn']
                    result.make = ai_result.get('make')
                    result.model = ai_result.get('model')
                    result.document_type = ai_result.get('document_type') or self.detect_document_type(full_text, filename)
                    result.extraction_method = "ai"
                    result.confidence = 0.70
                    result.success = True
                    
                    logger.info(f"✅ AI extraction successful: {result.vrn}")
                    return result
            
            # Step 6: No VRN found
            result.error = "VRN not found"
            logger.info(f"ℹ️ No VRN found in {filename}")
            
            return result
            
        except Exception as e:
            result.error = str(e)
            logger.error(f"❌ Error processing {filename}: {e}", exc_info=True)
            return result
    
    
    async def get_unassigned_documents(self) -> List[Dict[str, str]]:
        """
        Get list of documents that need VRN extraction.
        
        Returns documents where:
        - vehicle_id IS NULL
        - extracted_data->>'vrn' IS NULL or extracted_data = '{}'
        
        Returns:
            List of dicts with id, raw_file_path
        """
        try:
            conn = self._get_db_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            sql = """
            SELECT 
                id::text as id,
                raw_file_path
            FROM document_registry
            WHERE vehicle_id IS NULL
              AND (
                  extracted_data IS NULL 
                  OR extracted_data = '{}'::jsonb
                  OR extracted_data->>'vrn' IS NULL
              )
              AND raw_file_path IS NOT NULL
            ORDER BY uploaded_at DESC
            LIMIT 1000
            """
            
            cur.execute(sql)
            results = cur.fetchall()
            
            cur.close()
            conn.close()
            
            documents = [
                {
                    'id': row['id'],
                    'raw_file_path': row['raw_file_path']
                }
                for row in results
            ]
            
            logger.info(f"📋 Found {len(documents)} documents needing VRN extraction")
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error getting unassigned documents: {e}")
            return []
    
    
    async def update_document_registry(
        self, 
        document_id: str, 
        extracted_data: Dict[str, Optional[str]]
    ) -> bool:
        """
        Update document_registry with extracted VRN and vehicle info.
        
        Args:
            document_id: Document registry UUID
            extracted_data: Dict with vrn, make, model, document_type
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            
            # Build JSONB data
            # Only include non-null values
            jsonb_data = {}
            
            if extracted_data.get('vrn'):
                jsonb_data['vrn'] = extracted_data['vrn']
            
            if extracted_data.get('make'):
                jsonb_data['make'] = extracted_data['make']
            
            if extracted_data.get('model'):
                jsonb_data['model'] = extracted_data['model']
            
            if extracted_data.get('document_type'):
                jsonb_data['document_type'] = extracted_data['document_type']
            
            if not jsonb_data:
                logger.warning(f"⚠️ No data to update for document {document_id}")
                return False
            
            # Update document_registry
            import json
            
            sql = """
            UPDATE document_registry
            SET extracted_data = %s::jsonb,
                updated_at = NOW()
            WHERE id = %s::uuid
            """
            
            cur.execute(sql, (json.dumps(jsonb_data), document_id))
            conn.commit()
            
            rows_updated = cur.rowcount
            
            cur.close()
            conn.close()
            
            if rows_updated > 0:
                logger.info(f"✅ Updated document {document_id} with: {jsonb_data}")
                return True
            else:
                logger.warning(f"⚠️ No rows updated for document {document_id}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error updating document {document_id}: {e}")
            return False
    
    
    async def process_batch(
        self, 
        document_ids: Optional[List[str]] = None,
        use_ai: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple documents in batch.
        
        Args:
            document_ids: List of document IDs to process (None = all unassigned)
            use_ai: Whether to use AI for extraction
            
        Returns:
            Dict with statistics
        """
        logger.info("🚀 Starting batch VRN extraction...")
        
        # Get documents to process
        if document_ids:
            # Process specific documents
            conn = self._get_db_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            sql = """
            SELECT 
                id::text as id,
                raw_file_path
            FROM document_registry
            WHERE id = ANY(%s::uuid[])
              AND raw_file_path IS NOT NULL
            """
            
            cur.execute(sql, (document_ids,))
            documents = [
                {'id': row['id'], 'raw_file_path': row['raw_file_path']}
                for row in cur.fetchall()
            ]
            
            cur.close()
            conn.close()
        else:
            # Process all unassigned documents
            documents = await self.get_unassigned_documents()
        
        if not documents:
            logger.info("ℹ️ No documents to process")
            return {
                'total_processed': 0,
                'vrn_found': 0,
                'vrn_not_found': 0,
                'failed': 0
            }
        
        logger.info(f"📦 Processing {len(documents)} documents...")
        
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
        for i, doc in enumerate(documents, 1):
            try:
                logger.info(f"📄 Processing {i}/{len(documents)}: {doc['raw_file_path']}")
                
                # Extract VRN
                result = await self.process_document(
                    doc['id'],
                    doc['raw_file_path'],
                    use_ai=use_ai
                )
                
                stats['total_processed'] += 1
                
                if result.success and result.vrn:
                    # Update database
                    update_success = await self.update_document_registry(
                        doc['id'],
                        {
                            'vrn': result.vrn,
                            'make': result.make,
                            'model': result.model,
                            'document_type': result.document_type
                        }
                    )
                    
                    if update_success:
                        stats['vrn_found'] += 1
                        stats['extraction_methods'][result.extraction_method] += 1
                        logger.info(f"✅ {i}/{len(documents)}: VRN found - {result.vrn}")
                    else:
                        stats['failed'] += 1
                        logger.error(f"❌ {i}/{len(documents)}: Database update failed")
                else:
                    stats['vrn_not_found'] += 1
                    logger.info(f"ℹ️ {i}/{len(documents)}: No VRN found")
                
            except Exception as e:
                stats['failed'] += 1
                logger.error(f"❌ Error processing document {i}/{len(documents)}: {e}")
                continue
        
        logger.info("=" * 60)
        logger.info(f"✅ Batch processing completed!")
        logger.info(f"   Total processed: {stats['total_processed']}")
        logger.info(f"   VRN found: {stats['vrn_found']}")
        logger.info(f"   VRN not found: {stats['vrn_not_found']}")
        logger.info(f"   Failed: {stats['failed']}")
        logger.info(f"   Methods: {stats['extraction_methods']}")
        logger.info("=" * 60)
        
        return stats


# Singleton instance
_vrn_service = None


def get_vrn_extraction_service() -> VRNExtractionService:
    """Get singleton VRN extraction service instance"""
    global _vrn_service
    
    if _vrn_service is None:
        _vrn_service = VRNExtractionService()
    
    return _vrn_service