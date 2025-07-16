#!/bin/bash
echo "🚀 RAG System Performance Monitor"
echo "================================"

echo "📊 Memory Usage:"
free -h

echo ""
echo "🤖 Ollama Status:"
curl -s http://localhost:11434/api/tags | jq '.models[] | {name: .name, size: .size}'

echo ""
echo "🗄️ Database Stats:"
cd /opt/rag_indexer
source venv/bin/activate
python -c "
import psycopg2
from dotenv import load_dotenv
import os
load_dotenv()
conn = psycopg2.connect(os.getenv('SUPABASE_CONNECTION_STRING'))
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM vecs.documents')
print(f'📄 Total chunks in database: {cur.fetchone()[0]}')
cur.execute('SELECT COUNT(DISTINCT metadata->>\'file_name\') FROM vecs.documents')
print(f'📁 Unique documents: {cur.fetchone()[0]}')
cur.close()
conn.close()
"

echo ""
echo "⚡ Performance Tips:"
echo "   • Use llama3-fast model for best speed"
echo "   • Keep 'Final Sources' at 1-2 for fastest responses"
echo "   • Increase 'Similarity Threshold' if getting irrelevant results"
echo "   • Monitor this with: bash /opt/monitor_performance.sh"
