#!/usr/bin/env python3
# ====================================
# ????: backend/classic_rag_test.py
# ?????????? ?????????? ??? ???????????? ???????????? RAG ???????
# ====================================

"""
Classic RAG Test Console - ?????????? ?????? ???????????? RAG ???????
??????? ????????? ??? ????????? ? ??????? ??????? ????????
"""

import asyncio
import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# ????????? ?????????? ????????? ?? .env ?????
load_dotenv()

# ????????? ???? ??? ???????
current_dir = Path(__file__).parent
backend_dir = current_dir.parent / "backend" if current_dir.name == "streamlit-rag" else current_dir
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(backend_dir))

# ????????? ???????????
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ClassicRAGTester:
    """?????????? ?????? ???????????? RAG ???????"""
    
    def __init__(self):
        self.vector_service = None
        self.llm_service = None
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "avg_search_time": 0.0,
            "avg_llm_time": 0.0,
            "avg_total_time": 0.0
        }
        
        print("?? Classic RAG Test Console")
        print("=" * 50)
    
    async def initialize(self):
        """?????????????? ???????"""
        try:
            print("?? Initializing services...")
            
            # ?????????????? ????????? ?????? - ?????? ?????? ?? ??????? ?????
            try:
                from supabase_vector_service import SupabaseVectorService
            except ImportError as e:
                print(f"? Cannot import SupabaseVectorService: {e}")
                print("   Make sure supabase_vector_service.py is in the same directory")
                return False
            
            # ????????? ?????? ???????? ?????????? ?????????
            connection_string = (
                os.getenv("SUPABASE_CONNECTION_STRING") or 
                os.getenv("DATABASE_URL") or
                os.getenv("POSTGRES_URL")
            )
            
            if not connection_string:
                print("? Database connection string not found!")
                print("   Looking for environment variables:")
                print("   - SUPABASE_CONNECTION_STRING")
                print("   - DATABASE_URL") 
                print("   - POSTGRES_URL")
                print("\n?? Check your .env file or set environment variables")
                return False
            
            self.vector_service = SupabaseVectorService(
                connection_string=connection_string,
                table_name="documents"  # ?????????? ?? ?? ??????? ??? ? rag_indexer
            )
            
            # ?????????????? LLM ?????? - ?????? ?????? ?? ??????? ?????
            try:
                from simple_llm_service import create_simple_llm_service
            except ImportError as e:
                print(f"? Cannot import SimpleLLMService: {e}")
                print("   Make sure simple_llm_service.py is in the same directory")
                return False
            
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_model = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b")
            
            self.llm_service = create_simple_llm_service(
                ollama_url=ollama_url,
                model=ollama_model
            )
            
            # ????????? ?????? ????????
            print("?? Checking services status...")
            
            # ????????? ???? ??????
            try:
                db_stats = await self.vector_service.get_database_stats()
                print(f"   ? Database: {db_stats['total_documents']} documents, {db_stats['unique_files']} files")
            except Exception as e:
                print(f"   ? Database error: {e}")
                return False
            
            # ????????? LLM
            llm_available = await self.llm_service.check_availability()
            if llm_available:
                print(f"   ? LLM: {self.llm_service.model} available at {ollama_url}")
            else:
                print(f"   ?? LLM: {self.llm_service.model} not available at {ollama_url} (will use fallback)")
            
            print("? Initialization complete!")
            return True
            
        except Exception as e:
            print(f"? Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_query(self, question: str, language: str = "en") -> Dict:
        """
        ????????? ?????? ???? RAG ???????
        
        Args:
            question: ?????? ????????????
            language: ???? ??????
            
        Returns:
            ?????????? ??????? ? ?????????
        """
        total_start = time.time()
        
        print(f"\n?? Processing: '{question}'")
        print("-" * 50)
        
        try:
            # ???? 1: ????????? ?????
            print("1?? Vector search...")
            search_start = time.time()
            
            search_results = await self.vector_service.vector_search(
                query=question,
                limit=5,
                similarity_threshold=0.3
            )
            
            search_time = time.time() - search_start
            print(f"   ?? Search completed in {search_time:.3f}s")
            print(f"   ?? Found {len(search_results)} relevant documents")
            
            # ?????????? ????????? ?????????
            if search_results:
                print("   ?? Documents:")
                for i, result in enumerate(search_results, 1):
                    similarity = result.similarity_score
                    filename = result.filename
                    match_type = result.search_info["match_type"]
                    print(f"      {i}. {filename} (similarity: {similarity:.3f}, {match_type})")
            else:
                print("   ? No relevant documents found")
            
            # ???? 2: ????????? ??????
            print("\n2?? Generating answer...")
            llm_start = time.time()
            
            # ??????????? ?????????? ??? LLM ???????
            context_docs = []
            for result in search_results:
                context_docs.append({
                    'filename': result.filename,
                    'content': result.content,
                    'similarity_score': result.similarity_score
                })
            
            llm_response = await self.llm_service.generate_answer(
                question=question,
                context_docs=context_docs,
                language=language
            )
            
            llm_time = time.time() - llm_start
            total_time = time.time() - total_start
            
            print(f"   ?? LLM completed in {llm_time:.3f}s")
            
            # ????????? ??????????
            self._update_stats(search_time, llm_time, total_time, llm_response.success)
            
            # ??????????
            result = {
                "question": question,
                "language": language,
                "search_results": search_results,
                "llm_response": llm_response,
                "metrics": {
                    "search_time": search_time,
                    "llm_time": llm_time,
                    "total_time": total_time,
                    "documents_found": len(search_results),
                    "llm_success": llm_response.success
                }
            }
            
            return result
            
        except Exception as e:
            total_time = time.time() - total_start
            print(f"? Query failed: {e}")
            
            return {
                "question": question,
                "language": language,
                "error": str(e),
                "metrics": {
                    "total_time": total_time,
                    "documents_found": 0,
                    "llm_success": False
                }
            }
    
    def _update_stats(self, search_time: float, llm_time: float, total_time: float, success: bool):
        """????????? ?????????? ????????"""
        self.stats["total_queries"] += 1
        
        if success:
            self.stats["successful_queries"] += 1
        
        # ????????? ?????????? ???????
        n = self.stats["total_queries"]
        self.stats["avg_search_time"] = (self.stats["avg_search_time"] * (n-1) + search_time) / n
        self.stats["avg_llm_time"] = (self.stats["avg_llm_time"] * (n-1) + llm_time) / n
        self.stats["avg_total_time"] = (self.stats["avg_total_time"] * (n-1) + total_time) / n
    
    def print_result(self, result: Dict):
        """??????? ????????? ??????? ? ???????"""
        print("\n" + "=" * 50)
        print("?? RESULTS")
        print("=" * 50)
        
        if "error" in result:
            print(f"? Error: {result['error']}")
            return
        
        # ????? LLM
        llm_response = result["llm_response"]
        print("?? Answer:")
        print("-" * 20)
        
        if llm_response.success:
            print(llm_response.content)
        else:
            print(f"? LLM Error: {llm_response.error}")
            print("\nFallback response:")
            print(llm_response.content)
        
        # ???????
        metrics = result["metrics"]
        print(f"\n?? Performance:")
        print(f"   Search: {metrics['search_time']:.3f}s")
        print(f"   LLM: {metrics['llm_time']:.3f}s")
        print(f"   Total: {metrics['total_time']:.3f}s")
        print(f"   Documents: {metrics['documents_found']}")
        
        # ?????????
        search_results = result.get("search_results", [])
        if search_results:
            print(f"\n?? Sources ({len(search_results)}):")
            for i, doc in enumerate(search_results, 1):
                filename = doc.filename
                similarity = doc.similarity_score
                match_type = doc.search_info["match_type"]
                print(f"   {i}. {filename} ({similarity:.3f}, {match_type})")
    
    def print_stats(self):
        """??????? ????? ??????????"""
        print("\n" + "=" * 50)
        print("?? SESSION STATISTICS")
        print("=" * 50)
        
        stats = self.stats
        success_rate = (stats["successful_queries"] / stats["total_queries"] * 100) if stats["total_queries"] > 0 else 0
        
        print(f"Total queries: {stats['total_queries']}")
        print(f"Successful: {stats['successful_queries']} ({success_rate:.1f}%)")
        print(f"Average search time: {stats['avg_search_time']:.3f}s")
        print(f"Average LLM time: {stats['avg_llm_time']:.3f}s")
        print(f"Average total time: {stats['avg_total_time']:.3f}s")
    
    async def interactive_mode(self):
        """????????????? ????? ??????"""
        print("\n?? Interactive Mode")
        print("Commands:")
        print("  - Type your question to search")
        print("  - 'stats' to show statistics")
        print("  - 'info' to show service info")
        print("  - 'quit' or 'exit' to quit")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n? Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if question.lower() == 'stats':
                    self.print_stats()
                    continue
                
                if question.lower() == 'info':
                    await self.show_service_info()
                    continue
                
                # ?????????? ???? ?? ??????? (??????? ???????????)
                language = "uk" if any(char in question for char in "?????????????????????????????????") else "en"
                
                # ????????? ??????
                result = await self.run_query(question, language)
                self.print_result(result)
                
            except KeyboardInterrupt:
                print("\n\n?? Goodbye!")
                break
            except Exception as e:
                print(f"? Error: {e}")
    
    async def show_service_info(self):
        """?????????? ?????????? ? ????????"""
        print("\n?? Service Information:")
        print("-" * 30)
        
        # ?????????? ? ????????? ???????
        try:
            db_stats = await self.vector_service.get_database_stats()
            print("?? Vector Search Service:")
            print(f"   Database: {db_stats['table_name']}")
            print(f"   Documents: {db_stats['total_documents']}")
            print(f"   Unique files: {db_stats['unique_files']}")
            print(f"   Embedding model: {db_stats['embedding_model']}")
        except Exception as e:
            print(f"?? Vector Search Service: Error - {e}")
        
        # ?????????? ? LLM ???????
        try:
            llm_info = await self.llm_service.get_service_info()
            print(f"\n?? LLM Service:")
            print(f"   Model: {llm_info['model']}")
            print(f"   Available: {llm_info['available']}")
            print(f"   URL: {llm_info['ollama_url']}")
            print(f"   Features: {', '.join(llm_info['features'])}")
        except Exception as e:
            print(f"\n?? LLM Service: Error - {e}")
    
    async def benchmark_mode(self, test_queries: List[str]):
        """????? ????????? ? ????????? ?????????"""
        print(f"\n? Benchmark Mode - {len(test_queries)} queries")
        print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}/{len(test_queries)}] Testing: '{query}'")
            
            result = await self.run_query(query)
            
            # ??????? ????? ???????????
            metrics = result["metrics"]
            llm_response = result.get("llm_response")
            status = "?" if llm_response and llm_response.success else "?"
            
            print(f"   {status} {metrics['total_time']:.3f}s (search: {metrics['search_time']:.3f}s, llm: {metrics['llm_time']:.3f}s)")
            
            # ????????? ????? ????? ?????????
            await asyncio.sleep(0.5)
        
        print("\n?? Benchmark completed!")
        self.print_stats()

async def main():
    """??????? ???????"""
    # ?????????? ?????????? ? ????????????
    print("?? Configuration:")
    connection_string = (
        os.getenv("SUPABASE_CONNECTION_STRING") or 
        os.getenv("DATABASE_URL") or
        os.getenv("POSTGRES_URL")
    )
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") 
    ollama_model = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b")
    
    print(f"   Database: {'? Found' if connection_string else '? Not found'}")
    print(f"   Ollama URL: {ollama_url}")
    print(f"   Ollama Model: {ollama_model}")
    
    # ????????? ??????? ?????????? ?????????
    if not connection_string:
        print("\n? Error: Database connection string not found!")
        print("\n?? Create .env file in the streamlit-rag directory with:")
        print("   SUPABASE_CONNECTION_STRING=your_connection_string")
        print("   # or set DATABASE_URL=your_connection_string")
        print("\n?? Or export environment variable:")
        print("   export SUPABASE_CONNECTION_STRING='your_connection_string'")
        return
    
    # ??????? ? ?????????????? ??????
    tester = ClassicRAGTester()
    
    if not await tester.initialize():
        print("? Failed to initialize services. Exiting.")
        return
    
    # ????????? ????????? ????????? ??????
    if len(sys.argv) > 1:
        if sys.argv[1] == "benchmark":
            # ????? ?????????
            test_queries = [
                "John Nolan",
                "Breeda Daly training"
             ]
            await tester.benchmark_mode(test_queries)
        elif sys.argv[1] == "test":
            # ????????? ????
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "John Nolan"
            result = await tester.run_query(query)
            tester.print_result(result)
            tester.print_stats()
        else:
            print(f"? Unknown command: {sys.argv[1]}")
            print("Usage: python classic_rag_test.py [benchmark|test 'your question']")
    else:
        # ????????????? ?????
        await tester.interactive_mode()
        tester.print_stats()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n?? Classic RAG Test interrupted. Goodbye!")
    except Exception as e:
        print(f"\n? Fatal error: {e}")
        sys.exit(1)