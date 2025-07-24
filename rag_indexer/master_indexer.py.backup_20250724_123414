#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master RAG Document Indexer Controller
Manages processing of multiple subdirectories with Ollama restarts and organized logging

This script orchestrates the indexing process across multiple subdirectories:
- Scans for subdirectories in the specified root path
- Processes each subdirectory independently using the main indexer
- Restarts Ollama between directories to prevent memory leaks
- Maintains organized logs with clear directory separation
- Provides comprehensive progress tracking and error handling

Usage:
    python master_indexer.py

Configuration:
    Set MASTER_DOCUMENTS_DIR in .env file or modify the default path below
    The script will process all subdirectories found in the specified path
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv


def log_master_message(message, log_file_path="./logs/master_indexer.log"):
    """
    Log master indexer messages with timestamps
    
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


def restart_ollama_service():
    """
    Restart Ollama service to prevent memory leaks
    Includes proper wait time for service to fully restart
    """
    log_master_message("Restarting Ollama service to prevent memory leaks...")
    
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
        
        # Wait a moment for clean shutdown
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
            return False
        else:
            log_master_message("Ollama service started successfully")
        
        # Wait for Ollama to fully initialize (10 seconds as requested)
        log_master_message("Waiting 10 seconds for Ollama to fully initialize...")
        time.sleep(10)
        
        # Verify Ollama is responding
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                log_master_message("Ollama service is responding correctly")
                return True
            else:
                log_master_message(f"WARNING: Ollama responded with status {response.status_code}")
                return True  # Continue anyway
        except Exception as e:
            log_master_message(f"WARNING: Could not verify Ollama status: {e}")
            return True  # Continue anyway
        
    except subprocess.TimeoutExpired:
        log_master_message("ERROR: Ollama restart command timed out")
        return False
    except Exception as e:
        log_master_message(f"ERROR: Failed to restart Ollama: {e}")
        return False


def scan_subdirectories(root_path):
    """
    Scan for subdirectories in the specified root path
    
    Args:
        root_path: Root directory to scan for subdirectories
    
    Returns:
        list: List of subdirectory paths sorted by name
    """
    subdirectories = []
    
    try:
        root_path_obj = Path(root_path)
        
        if not root_path_obj.exists():
            log_master_message(f"ERROR: Root path does not exist: {root_path}")
            return []
        
        if not root_path_obj.is_dir():
            log_master_message(f"ERROR: Root path is not a directory: {root_path}")
            return []
        
        # Find all subdirectories
        for item in root_path_obj.iterdir():
            if item.is_dir():
                subdirectories.append(str(item))
        
        # Sort subdirectories by name for consistent processing order
        subdirectories.sort()
        
        log_master_message(f"Found {len(subdirectories)} subdirectories in {root_path}")
        for i, subdir in enumerate(subdirectories, 1):
            log_master_message(f"  {i}. {subdir}")
        
        return subdirectories
        
    except Exception as e:
        log_master_message(f"ERROR: Failed to scan subdirectories: {e}")
        return []


def process_single_directory(directory_path, directory_index, total_directories):
    """
    Process a single directory using the main indexer
    
    Args:
        directory_path: Path to the directory to process
        directory_index: Current directory index (1-based)
        total_directories: Total number of directories to process
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    directory_name = os.path.basename(directory_path)
    
    log_master_message(f"")
    log_master_message(f"{'='*80}")
    log_master_message(f"PROCESSING DIRECTORY {directory_index}/{total_directories}: {directory_name}")
    log_master_message(f"Path: {directory_path}")
    log_master_message(f"{'='*80}")
    
    # Create a temporary .env override for this directory
    original_env = os.environ.get('DOCUMENTS_DIR', '')
    os.environ['DOCUMENTS_DIR'] = directory_path
    
    # Add directory info to failed files log for organization
    failed_files_log = "./logs/failed_files_details.log"
    try:
        os.makedirs("./logs", exist_ok=True)
        with open(failed_files_log, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"PROCESSING DIRECTORY: {directory_path}\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directory {directory_index} of {total_directories}\n")
            f.write(f"{'='*80}\n")
    except Exception as e:
        log_master_message(f"WARNING: Could not write directory header to failed files log: {e}")
    
    try:
        # Run the main indexer for this directory
        start_time = time.time()
        
        log_master_message(f"Launching indexer.py for directory: {directory_name}")
        
        result = subprocess.run(
            [sys.executable, "indexer.py"],
            cwd=os.getcwd(),
            env=os.environ.copy(),
            capture_output=False,  # Let output go to console
            text=True
        )
        
        processing_time = time.time() - start_time
        
        if result.returncode == 0:
            log_master_message(f"SUCCESS: Directory {directory_name} processed successfully in {processing_time:.1f}s")
            return True
        else:
            log_master_message(f"ERROR: Directory {directory_name} processing failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        processing_time = time.time() - start_time if 'start_time' in locals() else 0
        log_master_message(f"ERROR: Exception while processing {directory_name}: {e}")
        return False
    
    finally:
        # Restore original environment
        if original_env:
            os.environ['DOCUMENTS_DIR'] = original_env
        elif 'DOCUMENTS_DIR' in os.environ:
            del os.environ['DOCUMENTS_DIR']


def create_final_summary(processed_directories, successful_directories, failed_directories, total_time):
    """
    Create and log final processing summary
    
    Args:
        processed_directories: Total directories processed
        successful_directories: Number of successful directories
        failed_directories: Number of failed directories
        total_time: Total processing time in seconds
    """
    log_master_message(f"")
    log_master_message(f"{'='*80}")
    log_master_message(f"MASTER INDEXER FINAL SUMMARY")
    log_master_message(f"{'='*80}")
    log_master_message(f"Total directories processed: {processed_directories}")
    log_master_message(f"Successful directories: {successful_directories}")
    log_master_message(f"Failed directories: {failed_directories}")
    log_master_message(f"Success rate: {(successful_directories/processed_directories*100):.1f}%" if processed_directories > 0 else "0%")
    log_master_message(f"Total processing time: {total_time/60:.1f} minutes")
    log_master_message(f"Average time per directory: {total_time/processed_directories:.1f} seconds" if processed_directories > 0 else "N/A")
    
    if failed_directories > 0:
        log_master_message(f"")
        log_master_message(f"WARNING: {failed_directories} directories failed to process")
        log_master_message(f"Check the master log for details on failed directories")
    else:
        log_master_message(f"")
        log_master_message(f"SUCCESS: All directories processed successfully!")
    
    log_master_message(f"")
    log_master_message(f"Detailed failed files information: ./logs/failed_files_details.log")
    log_master_message(f"Master processing log: ./logs/master_indexer.log")
    log_master_message(f"{'='*80}")


def main():
    """
    Main function to orchestrate multi-directory processing
    """
    print("Master RAG Document Indexer Controller")
    print("="*50)
    
    # Load environment variables
    load_dotenv()
    
    # Get root directory from environment or use default
    root_directory = os.getenv("MASTER_DOCUMENTS_DIR", "./data/634/2025")
    
    log_master_message(f"Master indexer started")
    log_master_message(f"Root directory: {root_directory}")
    
    # Scan for subdirectories
    subdirectories = scan_subdirectories(root_directory)
    
    if not subdirectories:
        log_master_message(f"ERROR: No subdirectories found in {root_directory}")
        print("ERROR: No subdirectories found to process")
        sys.exit(1)
    
    # Initialize counters
    total_directories = len(subdirectories)
    successful_directories = 0
    failed_directories = 0
    
    log_master_message(f"Starting processing of {total_directories} directories")
    
    master_start_time = time.time()
    
    # Process each subdirectory
    for index, directory_path in enumerate(subdirectories, 1):
        directory_name = os.path.basename(directory_path)
        
        try:
            # Process the directory
            success = process_single_directory(directory_path, index, total_directories)
            
            if success:
                successful_directories += 1
                log_master_message(f"Directory {directory_name} completed successfully")
            else:
                failed_directories += 1
                log_master_message(f"Directory {directory_name} failed")
            
            # Restart Ollama between directories (except after the last one)
            if index < total_directories:
                log_master_message(f"Preparing for next directory ({index + 1}/{total_directories})")
                
                restart_success = restart_ollama_service()
                if not restart_success:
                    log_master_message(f"WARNING: Ollama restart failed, but continuing...")
                
                log_master_message(f"Ready to process next directory")
            
        except KeyboardInterrupt:
            log_master_message(f"INTERRUPTED: Master indexer interrupted by user")
            log_master_message(f"Processed {successful_directories} directories successfully before interruption")
            print("\nMaster indexer interrupted by user")
            sys.exit(1)
        
        except Exception as e:
            failed_directories += 1
            log_master_message(f"FATAL ERROR processing {directory_name}: {e}")
    
    # Calculate total time
    total_time = time.time() - master_start_time
    
    # Create final summary
    create_final_summary(total_directories, successful_directories, failed_directories, total_time)
    
    # Print summary to console
    print(f"\nMaster indexer completed!")
    print(f"Processed: {total_directories} directories")
    print(f"Successful: {successful_directories}")
    print(f"Failed: {failed_directories}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Check ./logs/master_indexer.log for detailed information")
    
    # Exit with appropriate code
    if failed_directories > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMaster indexer interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR in master indexer: {e}")
        sys.exit(1)
