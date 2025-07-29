#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Master RAG Document Indexer Controller
Manages processing of multiple subdirectories with advanced features integration
ENHANCED: Full integration with new PDF processing, backup system, and safe restarts
FIXED: Backup directory correctly set to root level (./data/doc_backups)

This script orchestrates the indexing process across multiple subdirectories:
- Scans for subdirectories in the specified root path
- EXCLUDES service directories (doc_backups, logs, etc.) from processing
- Processes each subdirectory independently using the enhanced indexer
- Integrates with enhanced PDF processing and backup systems
- FIXED: Sets backup directory at root level for all subdirectories
- Manages safe Ollama restarts optimized with new batch restart system
- Maintains organized logs with comprehensive directory analysis
- Provides enhanced progress tracking and error handling with new features

Usage:
    python master_indexer.py

Configuration:
    Set MASTER_DOCUMENTS_DIR in .env file or modify the default path below
    The script will process all subdirectories found in the specified path
    EXCEPT service directories which contain backup/system files
    Backup directory will be set to parent/doc_backups for all subdirectories
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv


# Enhanced list of service directories to exclude from processing
EXCLUDED_DIRECTORIES = [
    'doc_backups',     # Enhanced backup directory for converted .doc files
    'logs',            # Log files directory
    'temp',            # Temporary files directory
    'cache',           # Cache files
    '.git',            # Git repository directory
    '__pycache__',     # Python cache directory
    '.vscode',         # VS Code settings
    '.idea',           # IntelliJ IDEA settings
    'node_modules',    # Node.js modules
    '.env',            # Environment files
    'backup',          # Generic backup directories
    'backups',         # Alternative backup naming
    'tmp',             # Alternative temp naming
    '.tmp'             # Hidden temp directories
]


def log_master_message(message, log_file_path="./logs/master_indexer.log"):
    """
    Enhanced log master indexer messages with timestamps and feature tracking
    
    Args:
        message: Message to log
        log_file_path: Path to master log file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    try:
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"WARNING: Could not write to master log: {e}")
    
    # Also print to console
    print(f"[MASTER] {message}")


def check_enhanced_features_availability():
    """
    Check availability of enhanced features (PDF processing, OCR, etc.)
    
    Returns:
        dict: Feature availability status
    """
    features = {
        'pdf_processing': {
            'available': False,
            'libraries': [],
            'missing': []
        },
        'ocr_processing': {
            'available': False,
            'libraries': [],
            'missing': []
        },
        'doc_conversion': {
            'available': False,
            'tools': [],
            'missing': []
        }
    }
    
    # Check PDF processing libraries
    pdf_libs = ['fitz', 'pdfplumber', 'pdf2image']
    for lib in pdf_libs:
        try:
            if lib == 'fitz':
                import fitz
            elif lib == 'pdfplumber':
                import pdfplumber
            elif lib == 'pdf2image':
                from pdf2image import convert_from_path
            features['pdf_processing']['libraries'].append(lib)
        except ImportError:
            features['pdf_processing']['missing'].append(lib)
    
    features['pdf_processing']['available'] = len(features['pdf_processing']['libraries']) > 0
    
    # Check OCR libraries
    ocr_libs = ['pytesseract', 'PIL', 'cv2']
    for lib in ocr_libs:
        try:
            if lib == 'pytesseract':
                import pytesseract
            elif lib == 'PIL':
                from PIL import Image
            elif lib == 'cv2':
                import cv2
            features['ocr_processing']['libraries'].append(lib)
        except ImportError:
            features['ocr_processing']['missing'].append(lib)
    
    features['ocr_processing']['available'] = len(features['ocr_processing']['libraries']) > 0
    
    # Check document conversion tools
    conversion_tools = [
        ('libreoffice', ['libreoffice', '--version']),
        ('pandoc', ['pandoc', '--version'])
    ]
    
    for tool_name, command in conversion_tools:
        try:
            result = subprocess.run(command, capture_output=True, timeout=10)
            if result.returncode == 0:
                features['doc_conversion']['tools'].append(tool_name)
        except:
            features['doc_conversion']['missing'].append(tool_name)
    
    features['doc_conversion']['available'] = len(features['doc_conversion']['tools']) > 0
    
    return features


def print_enhanced_features_status(features):
    """
    Print enhanced features availability status
    
    Args:
        features: Feature availability from check_enhanced_features_availability
    """
    log_master_message("ENHANCED FEATURES STATUS:")
    
    # PDF Processing
    pdf_status = "‚úÖ AVAILABLE" if features['pdf_processing']['available'] else "‚ùå LIMITED"
    log_master_message(f"  Ì†ΩÌ≥Ñ Enhanced PDF Processing: {pdf_status}")
    if features['pdf_processing']['libraries']:
        log_master_message(f"     Available: {', '.join(features['pdf_processing']['libraries'])}")
    if features['pdf_processing']['missing']:
        log_master_message(f"     Missing: {', '.join(features['pdf_processing']['missing'])}")
    
    # OCR Processing
    ocr_status = "‚úÖ AVAILABLE" if features['ocr_processing']['available'] else "‚ùå DISABLED"
    log_master_message(f"  Ì†ΩÌ¥ç OCR Processing: {ocr_status}")
    if features['ocr_processing']['libraries']:
        log_master_message(f"     Available: {', '.join(features['ocr_processing']['libraries'])}")
    if features['ocr_processing']['missing']:
        log_master_message(f"     Missing: {', '.join(features['ocr_processing']['missing'])}")
    
    # Document Conversion
    conv_status = "‚úÖ AVAILABLE" if features['doc_conversion']['available'] else "‚ùå LIMITED"
    log_master_message(f"  Ì†ΩÌ¥Ñ Document Conversion: {conv_status}")
    if features['doc_conversion']['tools']:
        log_master_message(f"     Available: {', '.join(features['doc_conversion']['tools'])}")
    if features['doc_conversion']['missing']:
        log_master_message(f"     Missing: {', '.join(features['doc_conversion']['missing'])}")


def is_excluded_directory(directory_path):
    """
    Enhanced check if directory should be excluded from processing
    
    Args:
        directory_path: Path to directory to check
    
    Returns:
        tuple: (is_excluded, reason)
    """
    directory_name = os.path.basename(directory_path).lower()
    
    # Check against enhanced excluded directories list
    for excluded in EXCLUDED_DIRECTORIES:
        if directory_name == excluded.lower():
            return True, f"Service directory ({excluded})"
    
    # Additional enhanced checks
    if directory_name.startswith('.'):
        return True, "Hidden directory"
    
    if not os.access(directory_path, os.R_OK):
        return True, "No read permission"
    
    if not os.access(directory_path, os.W_OK):
        return True, "No write permission (needed for .doc file deletion)"
    
    # Check if directory is empty
    try:
        if not any(os.scandir(directory_path)):
            return True, "Empty directory"
    except Exception:
        return True, "Cannot scan directory"
    
    # Enhanced check: look for backup directory patterns
    backup_patterns = ['backup', 'bak', 'old', 'archive', 'temp', 'tmp']
    if any(pattern in directory_name for pattern in backup_patterns):
        return True, f"Backup pattern directory"
    
    return False, None


def analyze_directory_content(directory_path):
    """
    Analyze directory content for enhanced processing insights
    
    Args:
        directory_path: Path to directory to analyze
    
    Returns:
        dict: Directory analysis results
    """
    analysis = {
        'total_files': 0,
        'doc_files': 0,
        'docx_files': 0,
        'pdf_files': 0,
        'image_files': 0,
        'text_files': 0,
        'estimated_processing_time': 0,
        'features_needed': [],
        'potential_issues': []
    }
    
    try:
        # Scan directory for file types
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                analysis['total_files'] += 1
                
                if file_ext == '.doc':
                    analysis['doc_files'] += 1
                    analysis['features_needed'].append('doc_conversion')
                elif file_ext == '.docx':
                    analysis['docx_files'] += 1
                elif file_ext == '.pdf':
                    analysis['pdf_files'] += 1
                    analysis['features_needed'].append('pdf_processing')
                elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                    analysis['image_files'] += 1
                    analysis['features_needed'].append('ocr_processing')
                elif file_ext in ['.txt', '.md', '.rst']:
                    analysis['text_files'] += 1
        
        # Remove duplicates from features needed
        analysis['features_needed'] = list(set(analysis['features_needed']))
        
        # Estimate processing time (rough)
        time_estimate = 0
        time_estimate += analysis['doc_files'] * 5      # 5 seconds per .doc conversion
        time_estimate += analysis['pdf_files'] * 10     # 10 seconds per PDF
        time_estimate += analysis['image_files'] * 8    # 8 seconds per image OCR
        time_estimate += analysis['total_files'] * 0.5  # 0.5 seconds per file processing
        
        analysis['estimated_processing_time'] = time_estimate
        
        # Identify potential issues
        if analysis['doc_files'] > 50:
            analysis['potential_issues'].append(f"Many .doc files ({analysis['doc_files']}) - conversion may take time")
        
        if analysis['pdf_files'] > 100:
            analysis['potential_issues'].append(f"Many PDF files ({analysis['pdf_files']}) - ensure PDF processing is optimized")
        
        if analysis['image_files'] > 200:
            analysis['potential_issues'].append(f"Many images ({analysis['image_files']}) - OCR processing will be intensive")
        
        if analysis['total_files'] > 10000:
            analysis['potential_issues'].append(f"Very large directory ({analysis['total_files']} files) - consider batch size optimization")
    
    except Exception as e:
        analysis['error'] = str(e)
    
    return analysis


def smart_restart_ollama_service(directory_index, total_directories, last_restart_time):
    """
    Smart Ollama restart that considers the new batch restart system
    
    Args:
        directory_index: Current directory index
        total_directories: Total number of directories
        last_restart_time: Time of last restart
    
    Returns:
        tuple: (restart_performed, new_last_restart_time)
    """
    current_time = time.time()
    
    # Don't restart if:
    # 1. It's the last directory
    # 2. Less than 300 seconds (5 minutes) since last restart
    # 3. We're processing small directories (let batch system handle it)
    
    if directory_index >= total_directories:
        log_master_message("Skipping Ollama restart - last directory")
        return False, last_restart_time
    
    if current_time - last_restart_time < 300:  # 5 minutes minimum interval
        log_master_message(f"Skipping Ollama restart - too recent ({(current_time - last_restart_time)/60:.1f}m ago)")
        return False, last_restart_time
    
    # Perform smart restart every 3-4 directories instead of every directory
    if directory_index % 3 != 0:
        log_master_message(f"Skipping Ollama restart - batch restart system will handle memory management")
        return False, last_restart_time
    
    log_master_message("Performing smart Ollama service restart...")
    log_master_message("INFO: Working with enhanced batch restart system (every 5 batches)")
    
    try:
        # Stop Ollama service
        stop_result = subprocess.run(
            ["sudo", "systemctl", "stop", "ollama"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if stop_result.returncode != 0:
            log_master_message(f"WARNING: Ollama stop command returned code {stop_result.returncode}")
        else:
            log_master_message("Ollama service stopped successfully")
        
        # Wait for clean shutdown
        time.sleep(3)
        
        # Start Ollama service
        start_result = subprocess.run(
            ["sudo", "systemctl", "start", "ollama"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if start_result.returncode != 0:
            log_master_message(f"ERROR: Ollama start command failed with code {start_result.returncode}")
            return False, last_restart_time
        else:
            log_master_message("Ollama service started successfully")
        
        # Wait for Ollama to fully initialize
        log_master_message("Waiting 10 seconds for Ollama to fully initialize...")
        time.sleep(10)
        
        # Verify Ollama is responding
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                log_master_message("‚úÖ Ollama service is responding correctly")
                return True, current_time
            else:
                log_master_message(f"‚ö†Ô∏è Ollama responded with status {response.status_code}")
                return True, current_time  # Continue anyway
        except Exception as e:
            log_master_message(f"‚ö†Ô∏è Could not verify Ollama status: {e}")
            return True, current_time  # Continue anyway
        
    except subprocess.TimeoutExpired:
        log_master_message("ERROR: Ollama restart command timed out")
        return False, last_restart_time
    except Exception as e:
        log_master_message(f"ERROR: Failed to restart Ollama: {e}")
        return False, last_restart_time


def scan_subdirectories(root_path):
    """
    Enhanced scan for subdirectories with content analysis
    
    Args:
        root_path: Root directory to scan for subdirectories
    
    Returns:
        tuple: (valid_subdirectories, excluded_subdirectories, directory_analyses)
    """
    valid_subdirectories = []
    excluded_subdirectories = []
    directory_analyses = {}
    
    try:
        root_path_obj = Path(root_path)
        
        if not root_path_obj.exists():
            log_master_message(f"ERROR: Root path does not exist: {root_path}")
            return [], [], {}
        
        if not root_path_obj.is_dir():
            log_master_message(f"ERROR: Root path is not a directory: {root_path}")
            return [], [], {}
        
        log_master_message(f"Enhanced scanning of root directory: {root_path}")
        
        # Find all subdirectories and analyze them
        for item in root_path_obj.iterdir():
            if item.is_dir():
                is_excluded, reason = is_excluded_directory(str(item))
                
                if is_excluded:
                    excluded_subdirectories.append((str(item), reason))
                    log_master_message(f"EXCLUDED: {item.name} - {reason}")
                else:
                    # Analyze directory content
                    analysis = analyze_directory_content(str(item))
                    directory_analyses[str(item)] = analysis
                    
                    if analysis.get('total_files', 0) > 0:
                        valid_subdirectories.append(str(item))
                        
                        # Log analysis summary
                        log_master_message(f"ANALYZED: {item.name}")
                        log_master_message(f"  Files: {analysis['total_files']} total")
                        if analysis['doc_files'] > 0:
                            log_master_message(f"  .doc files: {analysis['doc_files']} (conversion needed)")
                        if analysis['pdf_files'] > 0:
                            log_master_message(f"  PDF files: {analysis['pdf_files']} (enhanced processing)")
                        if analysis['image_files'] > 0:
                            log_master_message(f"  Images: {analysis['image_files']} (OCR processing)")
                        if analysis['estimated_processing_time'] > 60:
                            log_master_message(f"  Est. time: {analysis['estimated_processing_time']/60:.1f} minutes")
                        if analysis['potential_issues']:
                            for issue in analysis['potential_issues']:
                                log_master_message(f"  ‚ö†Ô∏è {issue}")
                    else:
                        excluded_subdirectories.append((str(item), "No files found"))
                        log_master_message(f"EXCLUDED: {item.name} - No files found")
        
        # Sort valid subdirectories by name for consistent processing order
        valid_subdirectories.sort()
        
        log_master_message(f"SCAN COMPLETE:")
        log_master_message(f"  Valid directories: {len(valid_subdirectories)}")
        log_master_message(f"  Excluded directories: {len(excluded_subdirectories)}")
        
        if valid_subdirectories:
            log_master_message("Valid directories for processing:")
            for i, subdir in enumerate(valid_subdirectories, 1):
                dir_name = os.path.basename(subdir)
                analysis = directory_analyses.get(subdir, {})
                files_info = f"{analysis.get('total_files', 0)} files"
                if analysis.get('estimated_processing_time', 0) > 60:
                    time_info = f", ~{analysis['estimated_processing_time']/60:.1f}m"
                else:
                    time_info = ""
                log_master_message(f"  {i}. {dir_name} ({files_info}{time_info})")
        
        return valid_subdirectories, excluded_subdirectories, directory_analyses
        
    except Exception as e:
        log_master_message(f"ERROR: Failed to scan subdirectories: {e}")
        return [], [], {}


def process_single_directory(directory_path, directory_index, total_directories, directory_analysis):
    """
    Enhanced processing of a single directory with feature integration
    
    Args:
        directory_path: Path to the directory to process
        directory_index: Current directory index (1-based)
        total_directories: Total number of directories to process
        directory_analysis: Analysis results for this directory
    
    Returns:
        tuple: (success, processing_stats)
    """
    directory_name = os.path.basename(directory_path)
    
    # Enhanced double-check for excluded directories
    is_excluded, reason = is_excluded_directory(directory_path)
    if is_excluded:
        log_master_message(f"SKIPPING: {directory_name} - {reason} (safety double-check)")
        return True, {'skipped': True, 'reason': reason}
    
    log_master_message(f"")
    log_master_message(f"{'='*80}")
    log_master_message(f"ENHANCED PROCESSING DIRECTORY {directory_index}/{total_directories}: {directory_name}")
    log_master_message(f"Path: {directory_path}")
    
    # Enhanced logging with content analysis
    if directory_analysis:
        log_master_message(f"Content Analysis:")
        log_master_message(f"  Total files: {directory_analysis.get('total_files', 0)}")
        if directory_analysis.get('doc_files', 0) > 0:
            log_master_message(f"  .doc files: {directory_analysis['doc_files']} (auto-conversion + backup + deletion)")
        if directory_analysis.get('pdf_files', 0) > 0:
            log_master_message(f"  PDF files: {directory_analysis['pdf_files']} (enhanced processing)")
        if directory_analysis.get('image_files', 0) > 0:
            log_master_message(f"  Images: {directory_analysis['image_files']} (OCR with auto-rotation)")
        
        features_needed = directory_analysis.get('features_needed', [])
        if features_needed:
            log_master_message(f"  Enhanced features needed: {', '.join(features_needed)}")
        
        est_time = directory_analysis.get('estimated_processing_time', 0)
        if est_time > 0:
            if est_time < 60:
                log_master_message(f"  Estimated processing time: {est_time:.0f} seconds")
            else:
                log_master_message(f"  Estimated processing time: {est_time/60:.1f} minutes")
    
    # FIXED: Log the backup directory that will be used
    backup_dir = os.getenv('DOC_BACKUP_ABSOLUTE_PATH', 'Not set')
    log_master_message(f"  Fixed backup directory: {backup_dir}")
    
    log_master_message(f"{'='*80}")
    
    # Create a temporary .env override for this directory
    original_env = os.environ.get('DOCUMENTS_DIR', '')
    os.environ['DOCUMENTS_DIR'] = directory_path
    
    # Enhanced failed files log organization
    failed_files_log = "./logs/failed_files_details.log"
    try:
        os.makedirs("./logs", exist_ok=True)
        with open(failed_files_log, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ENHANCED PROCESSING DIRECTORY: {directory_path}\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directory {directory_index} of {total_directories}\n")
            f.write(f"Fixed backup directory: {backup_dir}\n")
            if directory_analysis:
                f.write(f"Files: {directory_analysis.get('total_files', 0)} total\n")
                if directory_analysis.get('doc_files', 0) > 0:
                    f.write(f"  .doc files: {directory_analysis['doc_files']} (conversion + backup + deletion)\n")
                if directory_analysis.get('pdf_files', 0) > 0:
                    f.write(f"  PDF files: {directory_analysis['pdf_files']} (enhanced processing)\n")
                if directory_analysis.get('image_files', 0) > 0:
                    f.write(f"  Images: {directory_analysis['image_files']} (OCR processing)\n")
            f.write(f"{'='*80}\n")
    except Exception as e:
        log_master_message(f"WARNING: Could not write directory header to failed files log: {e}")
    
    processing_stats = {
        'directory_name': directory_name,
        'directory_path': directory_path,
        'success': False,
        'processing_time': 0,
        'analysis': directory_analysis,
        'features_used': [],
        'backup_directory': backup_dir
    }
    
    try:
        # Run the enhanced indexer for this directory
        start_time = time.time()
        
        log_master_message(f"Launching enhanced indexer.py for directory: {directory_name}")
        
        result = subprocess.run(
            [sys.executable, "indexer.py"],
            cwd=os.getcwd(),
            env=os.environ.copy(),
            capture_output=False,  # Let output go to console for real-time monitoring
            text=True
        )
        
        processing_time = time.time() - start_time
        processing_stats['processing_time'] = processing_time
        
        if result.returncode == 0:
            log_master_message(f"‚úÖ SUCCESS: Directory {directory_name} processed successfully in {processing_time:.1f}s")
            
            # Enhanced success logging
            if directory_analysis:
                files_processed = directory_analysis.get('total_files', 0)
                if files_processed > 0:
                    rate = files_processed / processing_time if processing_time > 0 else 0
                    log_master_message(f"Ì†ΩÌ≥ä Processing rate: {rate:.1f} files/second")
                
                # Log which enhanced features were likely used
                features_used = []
                if directory_analysis.get('doc_files', 0) > 0:
                    features_used.append('doc_conversion_with_backup_and_deletion')
                if directory_analysis.get('pdf_files', 0) > 0:
                    features_used.append('enhanced_pdf_processing')
                if directory_analysis.get('image_files', 0) > 0:
                    features_used.append('ocr_with_auto_rotation')
                
                if features_used:
                    log_master_message(f"Ì†ΩÌ∫Ä Enhanced features used: {', '.join(features_used)}")
                    processing_stats['features_used'] = features_used
            
            processing_stats['success'] = True
            return True, processing_stats
        else:
            log_master_message(f"‚ùå ERROR: Directory {directory_name} processing failed with return code {result.returncode}")
            processing_stats['error_code'] = result.returncode
            return False, processing_stats
            
    except Exception as e:
        processing_time = time.time() - start_time if 'start_time' in locals() else 0
        log_master_message(f"‚ùå EXCEPTION: Error processing {directory_name}: {e}")
        processing_stats['processing_time'] = processing_time
        processing_stats['error'] = str(e)
        return False, processing_stats
    
    finally:
        # Restore original environment
        if original_env:
            os.environ['DOCUMENTS_DIR'] = original_env
        elif 'DOCUMENTS_DIR' in os.environ:
            del os.environ['DOCUMENTS_DIR']


def create_enhanced_final_summary(processed_directories, successful_directories, failed_directories, 
                                excluded_directories, total_time, processing_stats_list, features_status):
    """
    Create enhanced final processing summary with feature usage analysis
    
    Args:
        processed_directories: Total directories processed
        successful_directories: Number of successful directories
        failed_directories: Number of failed directories
        excluded_directories: Number of excluded directories
        total_time: Total processing time in seconds
        processing_stats_list: List of processing statistics for each directory
        features_status: Enhanced features availability status
    """
    log_master_message(f"")
    log_master_message(f"{'='*80}")
    log_master_message(f"ENHANCED MASTER INDEXER FINAL SUMMARY")
    log_master_message(f"{'='*80}")
    
    # Basic statistics
    log_master_message(f"Ì†ΩÌ≥ä Processing Statistics:")
    log_master_message(f"  Total directories found: {processed_directories + excluded_directories}")
    log_master_message(f"  Directories processed: {processed_directories}")
    log_master_message(f"  Directories excluded: {excluded_directories} (service directories)")
    log_master_message(f"  Successful directories: {successful_directories}")
    log_master_message(f"  Failed directories: {failed_directories}")
    
    if processed_directories > 0:
        success_rate = (successful_directories / processed_directories * 100)
        log_master_message(f"  Success rate: {success_rate:.1f}%")
    else:
        log_master_message(f"  Success rate: 0%")
    
    log_master_message(f"  Total processing time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    if processed_directories > 0:
        avg_time = total_time / processed_directories
        log_master_message(f"  Average time per directory: {avg_time:.1f} seconds")
    
    # Enhanced feature usage analysis
    if processing_stats_list:
        log_master_message(f"")
        log_master_message(f"Ì†ΩÌ∫Ä Enhanced Features Usage Analysis:")
        
        # Aggregate feature usage
        feature_usage = {}
        total_files_by_type = {
            'doc_files': 0,
            'pdf_files': 0,
            'image_files': 0,
            'total_files': 0
        }
        
        for stats in processing_stats_list:
            if stats.get('success') and 'analysis' in stats and stats['analysis']:
                analysis = stats['analysis']
                total_files_by_type['doc_files'] += analysis.get('doc_files', 0)
                total_files_by_type['pdf_files'] += analysis.get('pdf_files', 0)
                total_files_by_type['image_files'] += analysis.get('image_files', 0)
                total_files_by_type['total_files'] += analysis.get('total_files', 0)
                
                for feature in stats.get('features_used', []):
                    feature_usage[feature] = feature_usage.get(feature, 0) + 1
        
        # Report file type processing
        if total_files_by_type['total_files'] > 0:
            log_master_message(f"  Ì†ΩÌ≥Å Files Processed by Type:")
            log_master_message(f"    Total files: {total_files_by_type['total_files']:,}")
            if total_files_by_type['doc_files'] > 0:
                log_master_message(f"    .doc files: {total_files_by_type['doc_files']:,} (converted + backed up + deleted)")
            if total_files_by_type['pdf_files'] > 0:
                log_master_message(f"    PDF files: {total_files_by_type['pdf_files']:,} (enhanced processing)")
            if total_files_by_type['image_files'] > 0:
                log_master_message(f"    Image files: {total_files_by_type['image_files']:,} (OCR with auto-rotation)")
        
        # Report feature usage
        if feature_usage:
            log_master_message(f"  Ì†ΩÌ¥ß Enhanced Features Used:")
            for feature, count in feature_usage.items():
                feature_display = feature.replace('_', ' ').title()
                log_master_message(f"    {feature_display}: {count} directories")
        
        # Performance insights
        successful_stats = [s for s in processing_stats_list if s.get('success')]
        if successful_stats:
            processing_times = [s['processing_time'] for s in successful_stats]
            fastest_time = min(processing_times)
            slowest_time = max(processing_times)
            avg_time = sum(processing_times) / len(processing_times)
            
            log_master_message(f"  ‚ö° Performance Insights:")
            log_master_message(f"    Fastest directory: {fastest_time:.1f}s")
            log_master_message(f"    Slowest directory: {slowest_time:.1f}s")
            log_master_message(f"    Average time: {avg_time:.1f}s")
            
            # Find fastest and slowest directories
            fastest_dir = min(successful_stats, key=lambda x: x['processing_time'])
            slowest_dir = max(successful_stats, key=lambda x: x['processing_time'])
            log_master_message(f"    Fastest: {os.path.basename(fastest_dir['directory_path'])}")
            log_master_message(f"    Slowest: {os.path.basename(slowest_dir['directory_path'])}")
    
    # Enhanced features availability summary
    log_master_message(f"")
    log_master_message(f"Ì†ΩÌ¥ß Enhanced Features Status:")
    if features_status['pdf_processing']['available']:
        log_master_message(f"  ‚úÖ Enhanced PDF Processing: Available")
        log_master_message(f"     Libraries: {', '.join(features_status['pdf_processing']['libraries'])}")
    else:
        log_master_message(f"  ‚ùå Enhanced PDF Processing: Limited")
        log_master_message(f"     Missing: {', '.join(features_status['pdf_processing']['missing'])}")
    
    if features_status['ocr_processing']['available']:
        log_master_message(f"  ‚úÖ OCR with Auto-rotation: Available")
        log_master_message(f"     Libraries: {', '.join(features_status['ocr_processing']['libraries'])}")
    else:
        log_master_message(f"  ‚ùå OCR Processing: Disabled")
        log_master_message(f"     Missing: {', '.join(features_status['ocr_processing']['missing'])}")
    
    if features_status['doc_conversion']['available']:
        log_master_message(f"  ‚úÖ Document Conversion with Backup: Available")
        log_master_message(f"     Tools: {', '.join(features_status['doc_conversion']['tools'])}")
        log_master_message(f"     Process: .doc ‚Üí .docx + backup + delete original")
    else:
        log_master_message(f"  ‚ùå Document Conversion: Limited")
        log_master_message(f"     Missing: {', '.join(features_status['doc_conversion']['missing'])}")
    
    # FIXED: Enhanced backup system information
    backup_dir = os.getenv('DOC_BACKUP_ABSOLUTE_PATH', 'Not configured')
    log_master_message(f"")
    log_master_message(f"Ì†ΩÌ≥Ç Fixed Backup System:")
    log_master_message(f"  ‚úÖ Backup directory: {backup_dir}")
    log_master_message(f"  ‚úÖ All subdirectories use the same backup location")
    log_master_message(f"  ‚úÖ Preserves original directory structure")
    log_master_message(f"  ‚úÖ Original .doc files deleted after successful backup")
    
    # Service directories information
    log_master_message(f"")
    log_master_message(f"Ì†ΩÌ≥Ç Service Directories Management:")
    if excluded_directories > 0:
        log_master_message(f"  ‚úÖ Excluded {excluded_directories} service directories:")
        log_master_message(f"     ‚Ä¢ doc_backups (backup files from .doc conversion)")
        log_master_message(f"     ‚Ä¢ logs (processing and error logs)")
        log_master_message(f"     ‚Ä¢ temp, cache, .git, __pycache__, etc.")
        log_master_message(f"  ‚úÖ This prevents processing backup/system files")
    else:
        log_master_message(f"  ‚ÑπÔ∏è No service directories found to exclude")
    
    # Error analysis
    if failed_directories > 0:
        log_master_message(f"")
        log_master_message(f"‚ö†Ô∏è Error Analysis:")
        log_master_message(f"  Failed directories: {failed_directories}")
        
        failed_stats = [s for s in processing_stats_list if not s.get('success')]
        if failed_stats:
            log_master_message(f"  Failed directory details:")
            for stats in failed_stats:
                dir_name = os.path.basename(stats.get('directory_path', 'Unknown'))
                error_info = stats.get('error', stats.get('error_code', 'Unknown error'))
                log_master_message(f"    ‚Ä¢ {dir_name}: {error_info}")
        
        log_master_message(f"  Check master log and individual directory logs for details")
        log_master_message(f"  Detailed failed files: ./logs/failed_files_details.log")
    else:
        log_master_message(f"")
        log_master_message(f"Ì†ºÌæâ SUCCESS: All valid directories processed successfully!")
        log_master_message(f"‚úÖ Enhanced backup system preserved all original .doc files")
        log_master_message(f"‚úÖ PDF processing extracted maximum text content")
        log_master_message(f"‚úÖ OCR auto-rotation optimized image text extraction")
    
    # System integration notes
    log_master_message(f"")
    log_master_message(f"Ì†ΩÌ¥ß System Integration Notes:")
    log_master_message(f"  ‚Ä¢ Safe Ollama restarts: Coordinated with batch-level restart system")
    log_master_message(f"  ‚Ä¢ Fixed backup system: {backup_dir}")
    log_master_message(f"  ‚Ä¢ Original cleanup: .doc files deleted after successful backup")
    log_master_message(f"  ‚Ä¢ PDF processing: Auto method selection for optimal extraction")
    log_master_message(f"  ‚Ä¢ OCR enhancement: Auto-rotation for improved text quality")
    log_master_message(f"  ‚Ä¢ Blacklist filtering: Automatic exclusion of service directories")
    
    log_master_message(f"")
    log_master_message(f"Ì†ΩÌ≥ã Logs and Reports:")
    log_master_message(f"  Master processing log: ./logs/master_indexer.log")
    log_master_message(f"  Detailed failed files: ./logs/failed_files_details.log")
    log_master_message(f"  Individual directory logs: ./logs/ (per-directory)")
    log_master_message(f"  Fixed backup location: {backup_dir}")
    log_master_message(f"{'='*80}")


def main():
    """
    Enhanced main function with full integration of new features and FIXED backup directory
    """
    print("Enhanced Master RAG Document Indexer Controller")
    print("=" * 60)
    print("Ì†ΩÌ∫Ä ENHANCED FEATURES:")
    print("  ‚Ä¢ Advanced PDF processing with auto method selection")
    print("  ‚Ä¢ Document conversion with structured backup + original deletion")
    print("  ‚Ä¢ OCR processing with auto-rotation optimization")
    print("  ‚Ä¢ Smart Ollama restart coordination with batch system")
    print("  ‚Ä¢ Automatic service directory exclusion")
    print("  ‚Ä¢ FIXED: Backup directory set at root level for all subdirectories")
    print("  ‚Ä¢ Comprehensive content analysis and progress tracking")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Get root directory from environment or use default
    root_directory = os.getenv("MASTER_DOCUMENTS_DIR", "./data/634/2025")
    
    # FIXED: Set fixed backup directory for all subdirectories
    if not os.getenv("DOC_BACKUP_ABSOLUTE_PATH"):
        master_backup_dir = os.path.join(os.path.dirname(root_directory), "doc_backups")
        os.environ['DOC_BACKUP_ABSOLUTE_PATH'] = master_backup_dir
        print(f"Ì†ΩÌ≥Ç Fixed backup directory: {master_backup_dir}")
    else:
        print(f"Ì†ΩÌ≥Ç Using configured backup directory: {os.getenv('DOC_BACKUP_ABSOLUTE_PATH')}")
    
    log_master_message(f"Enhanced Master Indexer started")
    log_master_message(f"Root directory: {root_directory}")
    log_master_message(f"Fixed backup directory: {os.getenv('DOC_BACKUP_ABSOLUTE_PATH', 'Not set')}")
    log_master_message(f"Service directories excluded: {', '.join(EXCLUDED_DIRECTORIES)}")
    
    # Check enhanced features availability
    features_status = check_enhanced_features_availability()
    print_enhanced_features_status(features_status)
    
    # Enhanced subdirectory scanning with content analysis
    valid_subdirectories, excluded_subdirectories, directory_analyses = scan_subdirectories(root_directory)
    
    if not valid_subdirectories:
        log_master_message(f"ERROR: No valid subdirectories found in {root_directory}")
        print("ERROR: No valid subdirectories found to process")
        print("All directories were either excluded (service directories) or contained no files")
        sys.exit(1)
    
    # Initialize enhanced counters
    total_valid_directories = len(valid_subdirectories)
    successful_directories = 0
    failed_directories = 0
    excluded_count = len(excluded_subdirectories)
    processing_stats_list = []
    
    log_master_message(f"Starting enhanced processing of {total_valid_directories} valid directories")
    log_master_message(f"Excluded {excluded_count} service directories from processing")
    
    # Enhanced features summary
    total_estimated_time = sum(
        directory_analyses.get(dir_path, {}).get('estimated_processing_time', 0) 
        for dir_path in valid_subdirectories
    )
    if total_estimated_time > 0:
        if total_estimated_time < 3600:
            log_master_message(f"Estimated total processing time: {total_estimated_time/60:.1f} minutes")
        else:
            log_master_message(f"Estimated total processing time: {total_estimated_time/3600:.1f} hours")
    
    master_start_time = time.time()
    last_restart_time = 0  # Track last Ollama restart for smart coordination
    
    # Process each valid subdirectory with enhanced features
    for index, directory_path in enumerate(valid_subdirectories, 1):
        directory_name = os.path.basename(directory_path)
        directory_analysis = directory_analyses.get(directory_path, {})
        
        try:
            # Enhanced processing with content analysis
            success, processing_stats = process_single_directory(
                directory_path, index, total_valid_directories, directory_analysis
            )
            
            processing_stats_list.append(processing_stats)
            
            if success:
                successful_directories += 1
                log_master_message(f"‚úÖ Directory {directory_name} completed successfully")
                
                # Enhanced success logging with feature usage
                if processing_stats.get('features_used'):
                    features_used = ', '.join(processing_stats['features_used'])
                    log_master_message(f"Ì†ΩÌ∫Ä Enhanced features used: {features_used}")
            else:
                failed_directories += 1
                log_master_message(f"‚ùå Directory {directory_name} failed")
            
            # Smart Ollama restart coordination (works with batch restart system)
            if index < total_valid_directories:
                log_master_message(f"Preparing for next directory ({index + 1}/{total_valid_directories})")
                
                restart_performed, last_restart_time = smart_restart_ollama_service(
                    index, total_valid_directories, last_restart_time
                )
                
                if restart_performed:
                    log_master_message(f"‚úÖ Smart Ollama restart completed")
                else:
                    log_master_message(f"‚ÑπÔ∏è Batch restart system will handle memory management")
                
                log_master_message(f"Ready to process next directory")
            
        except KeyboardInterrupt:
            log_master_message(f"INTERRUPTED: Enhanced master indexer interrupted by user")
            log_master_message(f"Processed {successful_directories} directories successfully before interruption")
            print("\nEnhanced master indexer interrupted by user")
            
            # Create partial summary
            total_time = time.time() - master_start_time
            create_enhanced_final_summary(
                index, successful_directories, failed_directories, 
                excluded_count, total_time, processing_stats_list, features_status
            )
            sys.exit(1)
        
        except Exception as e:
            failed_directories += 1
            log_master_message(f"FATAL ERROR processing {directory_name}: {e}")
            
            # Add error stats
            error_stats = {
                'directory_name': directory_name,
                'directory_path': directory_path,
                'success': False,
                'error': str(e),
                'processing_time': 0,
                'analysis': directory_analysis,
                'backup_directory': os.getenv('DOC_BACKUP_ABSOLUTE_PATH', 'Not set')
            }
            processing_stats_list.append(error_stats)
    
    # Calculate total time
    total_time = time.time() - master_start_time
    
    # Create enhanced final summary
    create_enhanced_final_summary(
        total_valid_directories, successful_directories, failed_directories, 
        excluded_count, total_time, processing_stats_list, features_status
    )
    
    # Enhanced console summary
    print(f"\nÌ†ºÌæØ Enhanced Master Indexer Completed!")
    print(f"Ì†ΩÌ≥ä Statistics:")
    print(f"  Total directories found: {total_valid_directories + excluded_count}")
    print(f"  Valid directories processed: {total_valid_directories}")
    print(f"  Service directories excluded: {excluded_count}")
    print(f"  Successful: {successful_directories}")
    print(f"  Failed: {failed_directories}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    
    # Enhanced features summary
    print(f"Ì†ΩÌ∫Ä Enhanced Features Used:")
    total_features_used = set()
    for stats in processing_stats_list:
        if stats.get('success') and stats.get('features_used'):
            total_features_used.update(stats['features_used'])
    
    if total_features_used:
        for feature in total_features_used:
            feature_display = feature.replace('_', ' ').title()
            print(f"  ‚úÖ {feature_display}")
    else:
        print(f"  ‚ÑπÔ∏è Standard processing used (no enhanced features needed)")
    
    # FIXED: Show backup directory information
    backup_dir = os.getenv('DOC_BACKUP_ABSOLUTE_PATH', 'Not configured')
    print(f"Ì†ΩÌ≥Ç Backup System:")
    print(f"  Fixed backup directory: {backup_dir}")
    print(f"  All .doc files backed up to same location")
    print(f"Ì†ΩÌ≥ã Logs: ./logs/master_indexer.log")
    
    # Exit with appropriate code
    if failed_directories > 0:
        print(f"‚ö†Ô∏è {failed_directories} directories failed - check logs for details")
        sys.exit(1)
    else:
        print(f"Ì†ºÌæâ All directories processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEnhanced master indexer interrupted by user")
        log_master_message("Enhanced master indexer interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR in enhanced master indexer: {e}")
        log_master_message(f"FATAL ERROR in enhanced master indexer: {e}")
        sys.exit(1)