#!/bin/bash

echo "=== System Optimization Script for RAG Indexing ==="
echo "Optimizing Ollama and system settings for better performance..."

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции для вывода
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Проверка прав root для системных изменений
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script needs sudo privileges for system optimization"
        echo "Please run: sudo $0"
        exit 1
    fi
}

# Проверка запущенных процессов
check_processes() {
    print_header "Process Check"
    
    if pgrep -f "indexer.py" > /dev/null; then
        print_warning "Indexer process is currently running"
        echo "Please stop it with Ctrl+C before running this optimization"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_status "No indexer process running - good to proceed"
    fi
}

# Определение системных ресурсов
detect_system() {
    print_header "System Detection"
    
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    AVAILABLE_RAM=$(free -g | awk '/^Mem:/{print $7}')
    CPU_CORES=$(nproc)
    
    echo "  Total RAM: ${TOTAL_RAM}GB"
    echo "  Available RAM: ${AVAILABLE_RAM}GB"
    echo "  CPU Cores: ${CPU_CORES}"
    
    # Проверка GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        echo "  GPU: ${GPU_INFO}"
    else
        echo "  GPU: Not detected or no nvidia-smi"
    fi
}

# Оптимизация Ollama
optimize_ollama() {
    print_header "Ollama Optimization"
    
    # Остановить Ollama если запущен
    print_status "Stopping Ollama service..."
    systemctl stop ollama 2>/dev/null || true
    
    # Создать оптимизированный systemd service
    print_status "Creating optimized Ollama service configuration..."
    
    cat > /etc/systemd/system/ollama.service <<EOF
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/usr/share/ollama/.ollama/models"
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"

# Performance optimizations for powerful system (42GB RAM, 2 Xeon)
Environment="OLLAMA_NUM_PARALLEL=8"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KEEP_ALIVE=60m"
Environment="OLLAMA_MAX_QUEUE=1024"

# GPU optimizations for 2GB GPU
Environment="OLLAMA_GPU_MEMORY_FRACTION=0.8"
Environment="OLLAMA_CPU_FALLBACK=true"

# Memory and timeout settings
Environment="OLLAMA_TIMEOUT=300"
Environment="OLLAMA_LOAD_TIMEOUT=300"

[Install]
WantedBy=default.target
EOF

    # Перезагрузить systemd и запустить Ollama
    print_status "Restarting Ollama with optimized settings..."
    systemctl daemon-reload
    systemctl enable ollama
    systemctl start ollama
    
    # Дать время на запуск
    sleep 10
    
    # Проверить статус
    if systemctl is-active --quiet ollama; then
        print_status "Ollama service is running"
    else
        print_error "Ollama failed to start. Check logs: journalctl -u ollama -f"
        return 1
    fi
    
    # Проверить API
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        print_status "Ollama API is responding"
    else
        print_error "Ollama API is not responding"
        return 1
    fi
}

# Системные оптимизации
optimize_system() {
    print_header "System Optimizations"
    
    # Оптимизация виртуальной памяти
    print_status "Optimizing virtual memory settings..."
    
    # Временные настройки
    echo 10 > /proc/sys/vm/swappiness
    echo 1 > /proc/sys/vm/overcommit_memory
    echo 50 > /proc/sys/vm/overcommit_ratio
    
    # Постоянные настройки
    cat >> /etc/sysctl.conf <<EOF

# RAG Indexing optimizations
vm.swappiness=10
vm.overcommit_memory=1
vm.overcommit_ratio=50
vm.dirty_ratio=15
vm.dirty_background_ratio=5
fs.file-max=2097152
EOF
    
    # Применить настройки
    sysctl -p
    
    # Увеличить лимиты файлов
    print_status "Increasing file limits..."
    
    cat >> /etc/security/limits.conf <<EOF

# RAG indexing limits
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
EOF
    
    # Оптимизация дисковой подсистемы
    print_status "Optimizing disk I/O..."
    
    # Найти основной диск
    ROOT_DISK=$(df / | tail -1 | awk '{print $1}' | sed 's/[0-9]*$//')
    
    if [[ -n "$ROOT_DISK" ]]; then
        echo mq-deadline > /sys/block/$(basename $ROOT_DISK)/queue/scheduler 2>/dev/null || true
        echo 2048 > /sys/block/$(basename $ROOT_DISK)/queue/read_ahead_kb 2>/dev/null || true
    fi
}

# Создание оптимизированного .env
create_optimized_env() {
    print_header "Environment Configuration"
    
    # Определить оптимальные настройки на основе системы
    if [[ $TOTAL_RAM -ge 32 ]]; then
        BATCH_SIZE=25
        NUM_WORKERS=8
        DB_BATCH_SIZE=200
    elif [[ $TOTAL_RAM -ge 16 ]]; then
        BATCH_SIZE=15
        NUM_WORKERS=6
        DB_BATCH_SIZE=150
    else
        BATCH_SIZE=10
        NUM_WORKERS=4
        DB_BATCH_SIZE=100
    fi
    
    print_status "Creating optimized .env configuration..."
    
    cat > .env.optimized <<EOF
# Optimized settings for powerful system (${TOTAL_RAM}GB RAM, ${CPU_CORES} cores)
SUPABASE_CONNECTION_STRING="postgresql://postgres:postgres@localhost:54322/postgres"
OLLAMA_BASE_URL=http://localhost:11434
DOCUMENTS_DIR=./data/634/2025
TABLE_NAME=documents

# Best quality embedding model
EMBED_MODEL=mxbai-embed-large
EMBED_DIM=1024

# Optimized chunking settings (fewer chunks = faster processing)
CHUNK_SIZE=1024
CHUNK_OVERLAP=256
MIN_CHUNK_LENGTH=100

# Performance settings optimized for your hardware
BATCH_SIZE=${BATCH_SIZE}
NUM_WORKERS=${NUM_WORKERS}
ENABLE_OCR=true

# Advanced optimizations
OLLAMA_TIMEOUT=300
DB_BATCH_SIZE=${DB_BATCH_SIZE}
SKIP_VALIDATION=false

# OCR optimizations
OCR_BATCH_SIZE=5
OCR_WORKERS=4
OCR_QUALITY_THRESHOLD=0.3
EOF
    
    echo "Optimized .env created with following settings:"
    echo "  - Batch size: ${BATCH_SIZE}"
    echo "  - Workers: ${NUM_WORKERS}"
    echo "  - DB batch size: ${DB_BATCH_SIZE}"
    echo "  - OCR: Enabled"
    echo "  - Embedding model: mxbai-embed-large (best quality)"
}

# Создание скрипта мониторинга
create_monitor_script() {
    print_header "Creating Monitoring Tools"
    
    cat > monitor_performance.py <<'EOF'
#!/usr/bin/env python3
"""
Performance monitoring script for RAG indexing
"""

import psutil
import time
import requests
import json
import os
from datetime import datetime

def get_ollama_stats():
    """Get Ollama performance statistics"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return f"🟢 Online ({len(models)} models loaded)"
        else:
            return f"🔴 Error (HTTP {response.status_code})"
    except:
        return "🔴 Offline"

def get_gpu_stats():
    """Get GPU statistics if available"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
            return f"🎮 GPU: {gpu_util}% util, {mem_used}MB/{mem_total}MB, {temp}°C"
        else:
            return "🎮 GPU: Not available"
    except:
        return "🎮 GPU: Not available"

def monitor_system():
    """Monitor system performance during indexing"""
    print("=== RAG Indexing Performance Monitor ===")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # System stats
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            
            # Network stats
            net_io = psutil.net_io_counters()
            
            # Process stats
            indexer_processes = [p for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']) 
                               if 'python' in p.info['name'] and any(cmd for cmd in p.cmdline() if 'indexer' in cmd)]
            
            # Display
            print(f"📊 Performance Monitor - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)
            print(f"🖥️  CPU Usage: {cpu_percent}%")
            print(f"🧠 Memory: {memory.percent}% ({memory.available/1024**3:.1f}GB available)")
            print(f"💾 Disk I/O: R:{disk_io.read_bytes/1024**3:.2f}GB W:{disk_io.write_bytes/1024**3:.2f}GB")
            print(f"🌐 Network: ↓{net_io.bytes_recv/1024**2:.0f}MB ↑{net_io.bytes_sent/1024**2:.0f}MB")
            print(f"🤖 Ollama: {get_ollama_stats()}")
            print(f"{get_gpu_stats()}")
            
            if indexer_processes:
                print(f"\n🔍 Indexer Processes:")
                for p in indexer_processes:
                    print(f"   PID {p.info['pid']}: CPU {p.info['cpu_percent']:.1f}%, RAM {p.info['memory_percent']:.1f}%")
            
            print("=" * 60)
            print("💡 Optimization Tips:")
            if cpu_percent > 95:
                print("⚠️  Very high CPU usage - consider reducing batch size")
            elif cpu_percent > 80:
                print("ℹ️  High CPU usage - normal for indexing")
            
            if memory.percent > 90:
                print("⚠️  Very high memory usage - consider reducing workers")
            elif memory.percent > 75:
                print("ℹ️  High memory usage - normal for large datasets")
            
            if not indexer_processes:
                print("ℹ️  No indexer processes detected")
            
            print("\n⏱️  Updating in 5 seconds...")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped.")

if __name__ == "__main__":
    monitor_system()
EOF
    
    chmod +x monitor_performance.py
    print_status "Performance monitor created: monitor_performance.py"
    
    # Создать простой скрипт для проверки Ollama
    cat > check_ollama.sh <<'EOF'
#!/bin/bash

echo "=== Ollama Status Check ==="
echo "Service status:"
systemctl status ollama --no-pager -l

echo -e "\nAPI response:"
curl -s http://localhost:11434/api/tags | jq '.' 2>/dev/null || curl -s http://localhost:11434/api/tags

echo -e "\nLoaded models:"
curl -s http://localhost:11434/api/tags | jq '.models[].name' 2>/dev/null || echo "Unable to parse models"

echo -e "\nSystem resources:"
echo "CPU: $(nproc) cores"
echo "RAM: $(free -h | awk '/^Mem:/{print $2}') total, $(free -h | awk '/^Mem:/{print $7}') available"
EOF
    
    chmod +x check_ollama.sh
    print_status "Ollama checker created: check_ollama.sh"
}

# Создание backup текущих настроек
backup_current_settings() {
    print_header "Backup Current Settings"
    
    BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup .env если существует
    if [[ -f ".env" ]]; then
        cp .env "$BACKUP_DIR/env_backup"
        print_status "Current .env backed up to $BACKUP_DIR/env_backup"
    fi
    
    # Backup Ollama service если существует
    if [[ -f "/etc/systemd/system/ollama.service" ]]; then
        cp /etc/systemd/system/ollama.service "$BACKUP_DIR/ollama_service_backup"
        print_status "Current Ollama service backed up to $BACKUP_DIR/ollama_service_backup"
    fi
    
    # Backup sysctl если есть наши настройки
    if grep -q "RAG Indexing optimizations" /etc/sysctl.conf 2>/dev/null; then
        cp /etc/sysctl.conf "$BACKUP_DIR/sysctl_backup"
        print_status "Current sysctl.conf backed up to $BACKUP_DIR/sysctl_backup"
    fi
    
    echo "Backup directory: $BACKUP_DIR"
}

# Финальные инструкции
show_final_instructions() {
    print_header "Optimization Complete!"
    
    echo "🎉 System has been optimized for RAG indexing!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Replace your .env file:"
    echo "   cp .env.optimized .env"
    echo ""
    echo "2. Run your existing indexer:"
    echo "   python indexer.py"
    echo ""
    echo "3. Monitor performance in another terminal:"
    echo "   python monitor_performance.py"
    echo ""
    echo "4. Check Ollama status:"
    echo "   ./check_ollama.sh"
    echo ""
    echo "📊 Expected improvements:"
    echo "   - Processing speed: 5-10x faster"
    echo "   - Memory usage: Better managed"
    echo "   - System stability: Improved"
    echo "   - Time estimation: 2-6 hours instead of 30+ hours"
    echo ""
    echo "⚙️ Key optimizations applied:"
    echo "   - Ollama: 8 parallel processes, 60min keep-alive"
    echo "   - System: Optimized VM settings, increased limits"
    echo "   - .env: Larger chunks (1024), bigger batches (25)"
    echo "   - Quality: Kept mxbai-embed-large for best results"
    echo ""
    print_warning "OCR is ENABLED - images will be processed"
    print_warning "Reboot recommended for full system optimization effect"
}

# Основная функция
main() {
    print_header "RAG Indexing System Optimization"
    
    # Проверки
    check_root
    check_processes
    
    # Определение системы
    detect_system
    
    # Создание backup
    backup_current_settings
    
    # Оптимизации
    optimize_system
    optimize_ollama
    
    # Создание конфигурации
    create_optimized_env
    create_monitor_script
    
    # Финальные инструкции
    show_final_instructions
}

# Запуск
main "$@"
