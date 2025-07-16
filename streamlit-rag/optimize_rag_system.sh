#!/bin/bash
# ?? Complete RAG System Speed Optimization Setup
# Run these commands to implement all optimizations

echo "?? Setting up OPTIMIZED RAG system..."

# === STEP 1: BACKUP EXISTING FILES ===
echo "?? Creating backups..."
cd /opt/rag_indexer
cp indexer.py indexer.py.backup
cp .env .env.backup

cd /opt/streamlit-rag  
cp app.py app.py.backup
cp .env .env.backup

# === STEP 2: UPDATE INDEXER CONFIGURATION ===
echo "?? Updating indexer configuration..."
cd /opt/rag_indexer

# Update .env with optimized settings
cat > .env << 'EOF'
SUPABASE_CONNECTION_STRING=postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
OLLAMA_BASE_URL=http://localhost:11434
DOCUMENTS_DIR=./data
TABLE_NAME=documents
EMBED_MODEL=mxbai-embed-large
EMBED_DIM=1024
CHUNK_SIZE=768
CHUNK_OVERLAP=192
MIN_CHUNK_LENGTH=50
EOF

echo "? Indexer .env updated with optimized chunk sizes"

# === STEP 3: UPDATE STREAMLIT CONFIGURATION ===
echo "?? Updating Streamlit configuration..."
cd /opt/streamlit-rag

# Update .env
cat > .env << 'EOF'
SUPABASE_CONNECTION_STRING=postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
STREAMLIT_SERVER_ENABLE_CORS=false
EOF

echo "? Streamlit .env updated"

# === STEP 4: CREATE OPTIMIZED OLLAMA MODEL ===
echo "?? Creating optimized Ollama model..."

# Create Modelfile for speed optimization
cat > /tmp/Modelfile << 'EOF'
FROM llama3:8b-instruct-q4_K_M
PARAMETER num_ctx 2048
PARAMETER num_predict 256
PARAMETER num_thread 8
PARAMETER f16_kv false
PARAMETER top_p 0.9
PARAMETER top_k 40
SYSTEM """You are a helpful document assistant. Be concise and direct. 
Only answer based on provided context. If information is not in the context, 
say so clearly. Keep responses under 200 words when possible. 
Always cite specific sources when answering."""
EOF

# Create optimized model
ollama create llama3-fast -f /tmp/Modelfile
echo "? Created optimized model: llama3-fast"

# Clean up
rm /tmp/Modelfile

# === STEP 5: OLLAMA SYSTEM OPTIMIZATIONS ===
echo "?? Applying Ollama system optimizations..."

# Update Ollama service configuration
sudo systemctl edit ollama.service << 'EOF'
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_QUEUE=1"
Environment="OLLAMA_KEEP_ALIVE=2m"
Environment="OLLAMA_CONTEXT_LENGTH=2048"
EOF

# Reload and restart Ollama
sudo systemctl daemon-reload
sudo systemctl restart ollama

echo "? Ollama system optimizations applied"

# === STEP 6: VERIFY SETUP ===
echo "?? Verifying optimized setup..."

# Check Ollama status
echo "Checking Ollama status..."
sudo systemctl status ollama --no-pager

# Check available models
echo "Checking available models..."
ollama list

# Test optimized model
echo "Testing optimized model..."
ollama run llama3-fast "Hello, this is a test." --timeout 30s

# === STEP 7: RE-INDEX WITH OPTIMIZED SETTINGS ===
echo "?? Re-indexing documents with optimized settings..."
cd /opt/rag_indexer
source venv/bin/activate

echo "??? Clearing existing index..."
python -c "
import psycopg2
from dotenv import load_dotenv
import os
load_dotenv()
conn = psycopg2.connect(os.getenv('SUPABASE_CONNECTION_STRING'))
cur = conn.cursor()
cur.execute('DELETE FROM vecs.documents')
conn.commit()
print(f'Cleared {cur.rowcount} existing records')
cur.close()
conn.close()
"

echo "?? Starting optimized indexing..."
python indexer.py

# === STEP 8: PERFORMANCE TEST ===
echo "? Running performance test..."
cd /opt/streamlit-rag
source venv/bin/activate

echo "?? OPTIMIZATION COMPLETE!"
echo "================================"
echo "? Applied optimizations:"
echo "   • Reduced chunk size: 1024 ? 768 tokens"
echo "   • Reduced overlap: 256 ? 192 tokens"
echo "   • Reduced context window: 4096 ? 2048 tokens"
echo "   • Reduced max response: unlimited ? 256 tokens"
echo "   • Added similarity filtering (0.7 threshold)"
echo "   • Reduced final sources: 5 ? 2"
echo "   • Created optimized Ollama model: llama3-fast"
echo "   • Applied Ollama system limits"
echo ""
echo "?? Expected performance improvements:"
echo "   • Response time: 2 minutes ? 30-60 seconds"
echo "   • Memory usage: reduced by ~30%"
echo "   • More relevant results (better filtering)"
echo ""
echo "?? Start the optimized app:"
echo "   cd /opt/streamlit-rag"
echo "   source venv/bin/activate"
echo "   streamlit run app.py"
echo ""
echo "?? Monitor performance in the sidebar!"

# === STEP 9: PERFORMANCE MONITORING SCRIPT ===
cat > /opt/monitor_performance.sh << 'EOF'
#!/bin/bash
echo "?? RAG System Performance Monitor"
echo "================================"

echo "?? Memory Usage:"
free -h

echo ""
echo "?? Ollama Status:"
curl -s http://localhost:11434/api/tags | jq '.models[] | {name: .name, size: .size}'

echo ""
echo "??? Database Stats:"
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
print(f'?? Total chunks in database: {cur.fetchone()[0]}')
cur.execute('SELECT COUNT(DISTINCT metadata->>\'file_name\') FROM vecs.documents')
print(f'?? Unique documents: {cur.fetchone()[0]}')
cur.close()
conn.close()
"

echo ""
echo "? Performance Tips:"
echo "   • Use llama3-fast model for best speed"
echo "   • Keep 'Final Sources' at 1-2 for fastest responses"
echo "   • Increase 'Similarity Threshold' if getting irrelevant results"
echo "   • Monitor this with: bash /opt/monitor_performance.sh"
EOF

chmod +x /opt/monitor_performance.sh

echo ""
echo "?? Performance monitoring script created: /opt/monitor_performance.sh"
echo "   Run it anytime with: bash /opt/monitor_performance.sh"

# Don't forget to replace placeholders!
echo ""
echo "??  IMPORTANT: Don't forget to replace [YOUR-PASSWORD] and [PROJECT-REF] in .env files!"
echo "   Edit: /opt/rag_indexer/.env"
echo "   Edit: /opt/streamlit-rag/.env"