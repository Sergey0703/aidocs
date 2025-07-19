# Simple LLM Service for Classic RAG System

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SimpleLLMResponse:
    """Simple LLM response structure"""
    content: str
    success: bool
    response_time: float
    tokens_used: int = 0
    model: str = "unknown"
    error: Optional[str] = None

class SimpleLLMService:
    """Simple LLM service focused on speed and simplicity"""
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 model: str = "llama3:8b-instruct-q4_K_M",
                 timeout: int = 60):
        """Initialize simple LLM service"""
        self.ollama_url = ollama_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.service_available = False
        
        logger.info(f"Simple LLM Service initialized: {self.model} @ {self.ollama_url}")
    
    async def check_availability(self) -> bool:
        """Quick check for Ollama availability"""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.ollama_url}/api/tags") as response:
                    self.service_available = response.status == 200
                    return self.service_available
        except:
            self.service_available = False
            return False
    
    async def generate_answer(self, 
                            question: str, 
                            context_docs: List[Dict] = None,
                            language: str = "en") -> SimpleLLMResponse:
        """Generate simple answer based on context"""
        start_time = time.time()
        
        # Check service availability
        if not self.service_available:
            available = await self.check_availability()
            if not available:
                return SimpleLLMResponse(
                    content=self._create_fallback_response(question, context_docs, language),
                    success=False,
                    response_time=time.time() - start_time,
                    error="Ollama service not available"
                )
        
        try:
            # Create simple prompt
            prompt = self._create_simple_prompt(question, context_docs, language)
            
            # Send request to Ollama
            response = await self._call_ollama(prompt)
            
            response_time = time.time() - start_time
            
            if response["success"]:
                logger.info(f"LLM response generated in {response_time:.2f}s ({len(response['content'])} chars)")
                
                return SimpleLLMResponse(
                    content=response["content"],
                    success=True,
                    response_time=response_time,
                    tokens_used=response.get("tokens", 0),
                    model=self.model
                )
            else:
                logger.warning(f"LLM request failed: {response['error']}")
                
                return SimpleLLMResponse(
                    content=self._create_fallback_response(question, context_docs, language),
                    success=False,
                    response_time=response_time,
                    error=response["error"]
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"LLM service error: {e}")
            
            return SimpleLLMResponse(
                content=self._create_fallback_response(question, context_docs, language),
                success=False,
                response_time=response_time,
                error=str(e)
            )
    
    def _create_simple_prompt(self, 
                            question: str, 
                            context_docs: List[Dict], 
                            language: str) -> str:
        """Create simplified prompt without complex instructions"""
        # Limit documents for speed
        max_docs = 2
        max_context_length = 800
        
        if not context_docs:
            # No context - very simple prompt
            return f"Question: {question}\n\nAnswer:"
        
        # With context - also simple
        context_parts = []
        for i, doc in enumerate(context_docs[:max_docs]):
            try:
                filename = doc.get('filename', f'Document {i+1}')
                content = doc.get('content', '')
                
                # Strongly truncate context for speed
                if len(content) > max_context_length:
                    content = content[:max_context_length] + "..."
                
                context_parts.append(f"Document {filename}:\n{content}")
                
            except Exception as e:
                logger.warning(f"Error processing context doc {i}: {e}")
                continue
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Documents:
{context}

Question: {question}

Brief answer based on documents:"""
        
        return prompt
    
    async def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama API with simple parameters"""
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            # Maximally simple payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for accuracy
                    "num_predict": 200,  # Short answers
                    "top_k": 10,
                    "top_p": 0.9
                }
            }
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate", 
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        content = data.get("response", "").strip()
                        tokens = data.get("eval_count", 0)
                        
                        return {
                            "success": True,
                            "content": content,
                            "tokens": tokens
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "content": "",
                            "error": f"HTTP {response.status}: {error_text}"
                        }
                        
        except asyncio.TimeoutError:
            return {
                "success": False,
                "content": "",
                "error": "Request timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "content": "",
                "error": str(e)
            }
    
    def _create_fallback_response(self, 
                                question: str, 
                                context_docs: List[Dict], 
                                language: str) -> str:
        """Create fallback response when LLM is unavailable"""
        if not context_docs:
            return f"""AI unavailable, no documents found

Your question: "{question}"

Recommendations:
- Try rephrasing your query  
- Check Ollama service status
- Add relevant documents to database"""
        
        # Show found documents even without AI
        docs_info = []
        for i, doc in enumerate(context_docs[:3], 1):
            filename = doc.get('filename', f'Document {i}')
            content = doc.get('content', '')
            preview = content[:200] + "..." if len(content) > 200 else content
            
            docs_info.append(f"{i}. {filename}\n   {preview}")
        
        docs_text = "\n\n".join(docs_info)
        
        return f"""AI unavailable, but found {len(context_docs)} documents

Your question: "{question}"

Found documents:
{docs_text}

Start Ollama service for full AI responses"""
    
    async def get_service_info(self) -> Dict[str, Any]:
        """Return service information"""
        available = await self.check_availability()
        
        return {
            "service_name": "Simple LLM Service",
            "model": self.model,
            "ollama_url": self.ollama_url,
            "available": available,
            "timeout": self.timeout,
            "features": [
                "Fast responses",
                "Simple prompts", 
                "Minimal context",
                "Fallback support"
            ]
        }

# Factory function for convenience
def create_simple_llm_service(ollama_url: str = "http://localhost:11434",
                            model: str = "llama3.2:3b") -> SimpleLLMService:
    """Create simple LLM service"""
    return SimpleLLMService(ollama_url=ollama_url, model=model)