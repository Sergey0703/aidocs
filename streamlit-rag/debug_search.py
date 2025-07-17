from supabase_vector_service import SupabaseVectorService
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def debug_search():
    conn_string = os.getenv("SUPABASE_CONNECTION_STRING")
    service = SupabaseVectorService(conn_string)
    
    # Поиск с большим limit и низким threshold
    results = await service.vector_search(
        query="John Nolan", 
        limit=20,  # Больше результатов
        similarity_threshold=0.1  # Ниже порог
    )
    
    print(f"Total results: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.filename} (similarity: {result.similarity_score:.3f})")

asyncio.run(debug_search())
