#!/bin/bash
echo "?? RAG System Performance Monitor"
echo "================================"

echo "?? Memory Usage:"
free -h

echo ""
echo "?? Ollama Status:"
curl -s http://localhost:11434/api/tags 2>/dev/null | head -20

echo ""
echo "?? Available Models:"
ollama list

echo ""
echo "??? Database Stats:"
cd /opt/rag_indexer
source venv/bin/activate
python -c "
import psycopg2
from dotenv import load_dotenv
import os
load_dotenv()
try:
    conn = psycopg2.connect(os.getenv('SUPABASE_CONNECTION_STRING'))
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM vecs.documents')
    print(f'?? Total chunks in database: {cur.fetchone()[0]}')
    cur.execute('SELECT COUNT(DISTINCT metadata->>\'file_name\') FROM vecs.documents WHERE metadata->>\'file_name\' IS NOT NULL')
    print(f'?? Unique documents: {cur.fetchone()[0]}')
    cur.close()
    conn.close()
except Exception as e:
    print(f'Database connection error: {e}')
" 2>/dev/null

echo ""
echo "? Performance Tips:"
echo "   • Use llama3-fast model for best speed"
echo "   • Keep 'Final Sources' at 1-2 for fastest responses"
echo "   • Increase 'Similarity Threshold' if getting irrelevant results"
echo "   • Monitor this with: bash /opt/streamlit-rag/monitor_performance.sh"

echo ""
echo "?? Current Optimizations Active:"
echo "   • Chunk size: 768 tokens (reduced from 1024)"
echo "   • Context window: 2048 tokens (reduced from 4096)"
echo "   • Max response: 256 tokens (limited)"
echo "   • Similarity threshold: 0.7 (smart filtering)"
echo "   • Final sources: 2 (reduced from 5)"
