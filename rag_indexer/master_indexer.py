#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Two-Level Master RAG Document Indexer Controller
Dynamically discovers year directories and processes numbered subdirectories within each
FULLY DYNAMIC: Discovers whatever year/number structure actually exists

This script:
- Scans root directory and dynamically discovers ANY year directories (2015, 2018, 2025, etc.)
- Within each discovered year, scans for numbered subdirectories (1, 2, 3, etc.)
- EXCLUDES service directories (doc_backups, logs, etc.) from processing
- For each numbered subdirectory, sets DOCUMENTS_DIR and calls indexer.py
- Processes: ./data/634/YYYY/N for whatever YYYY and N actually exist

Usage:
   python master_indexer.py

Configuration:
   Set MASTER_DOCUMENTS_DIR in .env file or modify the default path below
   Example discovered structure:
   ./data/634/
   ‚îú‚îÄ‚îÄ 2018/
   ‚îÇ   ‚îú‚îÄ‚îÄ 1/     ‚Üê Process this
   ‚îÇ   ‚îî‚îÄ‚îÄ 3/     ‚Üê Then this (2 doesn't exist)
   ‚îú‚îÄ‚îÄ 2023/
   ‚îÇ   ‚îú‚îÄ‚îÄ 1/     ‚Üê Process this
   ‚îÇ   ‚îú‚îÄ‚îÄ 2/     ‚Üê Then this
   ‚îÇ   ‚îî‚îÄ‚îÄ 5/     ‚Üê Then this
   ‚îú‚îÄ‚îÄ 2025/
   ‚îÇ   ‚îî‚îÄ‚îÄ 1/     ‚Üê Process this
   ‚îî‚îÄ‚îÄ doc_backups/  ‚Üê Exclude this
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv


# Service directories to exclude from processing
EXCLUDED_DIRECTORIES = [
   'doc_backups',     # Backup directory for converted .doc files
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


def is_excluded_directory(directory_name):
   """
   Check if directory should be excluded from processing
   
   Args:
       directory_name: Name of directory to check
   
   Returns:
       tuple: (is_excluded, reason)
   """
   directory_name_lower = directory_name.lower()
   
   # Check against excluded directories list
   for excluded in EXCLUDED_DIRECTORIES:
       if directory_name_lower == excluded.lower():
           return True, f"Service directory ({excluded})"
   
   # Check for backup directory patterns
   backup_patterns = ['backup', 'bak', 'old', 'archive', 'temp', 'tmp']
   if any(pattern in directory_name_lower for pattern in backup_patterns):
       return True, f"Backup pattern directory"
   
   # Hidden directories
   if directory_name.startswith('.'):
       return True, "Hidden directory"
   
   return False, None


def has_files_in_directory(directory_path):
   """
   Quick check if directory has any files
   
   Args:
       directory_path: Path to directory to check
   
   Returns:
       tuple: (has_files, file_count)
   """
   try:
       file_count = 0
       for root, dirs, files in os.walk(directory_path):
           file_count += len(files)
           # Quick check - if we find files, we can return early
           if file_count > 0:
               return True, file_count
           # Don't go too deep for performance
           if file_count > 10:
               break
       return file_count > 0, file_count
   except Exception:
       return False, 0


def discover_year_directories(root_path):
   """
   Dynamically discover all year directories in root path
   
   Args:
       root_path: Root directory to scan (e.g., ./data/634)
   
   Returns:
       list: List of discovered year directory paths
   """
   year_directories = []
   
   try:
       root_path_obj = Path(root_path)
       
       if not root_path_obj.exists():
           log_master_message(f"ERROR: Root path does not exist: {root_path}")
           return []
       
       log_master_message(f"Dynamically discovering year directories in: {root_path}")
       
       # Scan for any directories that could be years or other valid containers
       for item in root_path_obj.iterdir():
           if item.is_dir():
               directory_name = item.name
               
               # Check if directory should be excluded
               is_excluded, reason = is_excluded_directory(directory_name)
               if is_excluded:
                   log_master_message(f"EXCLUDED YEAR DIR: {directory_name} - {reason}")
                   continue
               
               # Check if directory is accessible
               if not os.access(str(item), os.R_OK):
                   log_master_message(f"EXCLUDED YEAR DIR: {directory_name} - No read permission")
                   continue
               
               # Add to year directories list
               year_directories.append(str(item))
               log_master_message(f"DISCOVERED YEAR DIR: {directory_name}")
       
       # Sort year directories for consistent processing order
       year_directories.sort()
       
       log_master_message(f"YEAR DISCOVERY COMPLETE:")
       log_master_message(f"  Total year directories found: {len(year_directories)}")
       
       if year_directories:
           log_master_message("  Year directories discovered:")
           for i, year_dir in enumerate(year_directories, 1):
               year_name = os.path.basename(year_dir)
               log_master_message(f"    {i}. {year_name}")
       
       return year_directories
       
   except Exception as e:
       log_master_message(f"ERROR: Failed to discover year directories: {e}")
       return []


def discover_numbered_subdirectories(year_directory_path):
   """
   Dynamically discover numbered subdirectories within a year directory
   
   Args:
       year_directory_path: Path to year directory
   
   Returns:
       list: List of numbered subdirectory paths with files
   """
   numbered_subdirectories = []
   year_name = os.path.basename(year_directory_path)
   
   try:
       year_path_obj = Path(year_directory_path)
       
       log_master_message(f"  Scanning year {year_name} for numbered subdirectories...")
       
       # Scan for subdirectories
       for item in year_path_obj.iterdir():
           if item.is_dir():
               subdir_name = item.name
               
               # Check if subdirectory should be excluded
               is_excluded, reason = is_excluded_directory(subdir_name)
               if is_excluded:
                   log_master_message(f"    EXCLUDED: {year_name}/{subdir_name} - {reason}")
                   continue
               
               # Check if subdirectory is accessible
               if not os.access(str(item), os.R_OK):
                   log_master_message(f"    EXCLUDED: {year_name}/{subdir_name} - No read permission")
                   continue
               
               # Check if subdirectory has files
               has_files, file_count = has_files_in_directory(str(item))
               if not has_files:
                   log_master_message(f"    EXCLUDED: {year_name}/{subdir_name} - No files found")
                   continue
               
               # Add to numbered subdirectories list
               numbered_subdirectories.append(str(item))
               log_master_message(f"    FOUND: {year_name}/{subdir_name} ({file_count} files)")
       
       # Sort for consistent processing order (numeric sort if possible)
       def sort_key(path):
           name = os.path.basename(path)
           # Try to sort numerically, fall back to string sort
           try:
               return int(name)
           except ValueError:
               return name
       
       numbered_subdirectories.sort(key=sort_key)
       
       log_master_message(f"  Year {year_name} scan complete: {len(numbered_subdirectories)} valid subdirectories")
       
       return numbered_subdirectories
       
   except Exception as e:
       log_master_message(f"ERROR: Failed to scan year directory {year_directory_path}: {e}")
       return []


def discover_all_processing_directories(root_path):
   """
   Dynamically discover all directories to process (Year/Number structure)
   
   Args:
       root_path: Root directory to scan
   
   Returns:
       list: List of all numbered subdirectory paths to process
   """
   all_processing_directories = []
   
   # Step 1: Discover year directories
   year_directories = discover_year_directories(root_path)
   
   if not year_directories:
       log_master_message(f"No year directories found in {root_path}")
       return []
   
   # Step 2: For each year, discover numbered subdirectories
   for year_directory in year_directories:
       year_name = os.path.basename(year_directory)
       log_master_message(f"Discovering subdirectories in year: {year_name}")
       
       numbered_subdirs = discover_numbered_subdirectories(year_directory)
       all_processing_directories.extend(numbered_subdirs)
   
   log_master_message(f"")
   log_master_message(f"COMPLETE DISCOVERY SUMMARY:")
   log_master_message(f"  Years scanned: {len(year_directories)}")
   log_master_message(f"  Total processing directories found: {len(all_processing_directories)}")
   
   if all_processing_directories:
       log_master_message(f"  All directories to process:")
       for i, proc_dir in enumerate(all_processing_directories, 1):
           # Create readable path (Year/Number)
           parts = Path(proc_dir).parts
           if len(parts) >= 2:
               display_path = f"{parts[-2]}/{parts[-1]}"
           else:
               display_path = os.path.basename(proc_dir)
           log_master_message(f"    {i}. {display_path}")
   
   return all_processing_directories


def process_single_directory(directory_path, directory_index, total_directories):
   """
   Process a single numbered directory by calling indexer.py
   
   Args:
       directory_path: Path to the numbered directory to process
       directory_index: Current directory index (1-based)
       total_directories: Total number of directories to process
   
   Returns:
       tuple: (success, processing_time, error_message)
   """
   # Create readable directory identifier (Year/Number)
   parts = Path(directory_path).parts
   if len(parts) >= 2:
       directory_identifier = f"{parts[-2]}/{parts[-1]}"
   else:
       directory_identifier = os.path.basename(directory_path)
   
   log_master_message(f"")
   log_master_message(f"{'='*60}")
   log_master_message(f"PROCESSING DIRECTORY {directory_index}/{total_directories}: {directory_identifier}")
   log_master_message(f"Full path: {directory_path}")
   log_master_message(f"{'='*60}")
   
   # Create environment for subprocess
   env = os.environ.copy()
   env['DOCUMENTS_DIR'] = directory_path
   
   # Log the backup directory that will be used
   backup_dir = env.get('DOC_BACKUP_ABSOLUTE_PATH', 'Default (parent/doc_backups)')
   log_master_message(f"Backup directory: {backup_dir}")
   
   start_time = time.time()
   error_message = None
   
   try:
       log_master_message(f"Launching indexer.py for directory: {directory_identifier}")
       
       # Run indexer.py as subprocess
       result = subprocess.run(
           [sys.executable, "indexer.py"],
           cwd=os.getcwd(),
           env=env,
           capture_output=True,  # Capture output for logging
           text=True,
           timeout=3600  # 1 hour timeout per directory
       )
       
       processing_time = time.time() - start_time
       
       # Log the output from indexer.py (abbreviated)
       if result.stdout:
           stdout_lines = result.stdout.split('\n')
           important_lines = [line for line in stdout_lines if any(keyword in line.lower() for keyword in 
                            ['success', 'error', 'failed', 'completed', 'processed', 'warning'])]
           
           if important_lines:
               log_master_message(f"INDEXER KEY OUTPUT for {directory_identifier}:")
               for line in important_lines[-10:]:  # Last 10 important lines
                   if line.strip():
                       log_master_message(f"  {line}")
       
       if result.stderr:
           stderr_lines = result.stderr.split('\n')
           for line in stderr_lines[-5:]:  # Last 5 error lines
               if line.strip():
                   log_master_message(f"  ERROR: {line}")
       
       if result.returncode == 0:
           log_master_message(f"‚úÖ SUCCESS: Directory {directory_identifier} processed successfully in {processing_time:.1f}s")
           return True, processing_time, None
       else:
           error_message = f"indexer.py returned code {result.returncode}"
           log_master_message(f"‚ùå ERROR: Directory {directory_identifier} processing failed: {error_message}")
           return False, processing_time, error_message
           
   except subprocess.TimeoutExpired:
       processing_time = time.time() - start_time
       error_message = f"Processing timed out after {processing_time/60:.1f} minutes"
       log_master_message(f"‚ùå TIMEOUT: Directory {directory_identifier} processing timed out")
       return False, processing_time, error_message
       
   except Exception as e:
       processing_time = time.time() - start_time
       error_message = str(e)
       log_master_message(f"‚ùå EXCEPTION: Error processing {directory_identifier}: {e}")
       return False, processing_time, error_message


def create_final_summary(total_directories, successful_directories, failed_directories, 
                       total_time, processing_details, year_directories_found):
   """
   Create final processing summary
   
   Args:
       total_directories: Total directories processed
       successful_directories: Number of successful directories
       failed_directories: Number of failed directories
       total_time: Total processing time in seconds
       processing_details: List of processing details for each directory
       year_directories_found: Number of year directories discovered
   """
   log_master_message(f"")
   log_master_message(f"{'='*60}")
   log_master_message(f"DYNAMIC MASTER INDEXER FINAL SUMMARY")
   log_master_message(f"{'='*60}")
   
   # Discovery statistics
   log_master_message(f"Ì†ΩÌ¥ç Dynamic Discovery Results:")
   log_master_message(f"  Year directories discovered: {year_directories_found}")
   log_master_message(f"  Numbered directories found: {total_directories}")
   
   # Processing statistics
   log_master_message(f"Ì†ΩÌ≥ä Processing Statistics:")
   log_master_message(f"  Directories processed: {total_directories}")
   log_master_message(f"  Successful directories: {successful_directories}")
   log_master_message(f"  Failed directories: {failed_directories}")
   
   if total_directories > 0:
       success_rate = (successful_directories / total_directories * 100)
       log_master_message(f"  Success rate: {success_rate:.1f}%")
   else:
       log_master_message(f"  Success rate: 0%")
   
   log_master_message(f"  Total processing time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
   if total_directories > 0:
       avg_time = total_time / total_directories
       log_master_message(f"  Average time per directory: {avg_time:.1f} seconds")
   
   # Performance insights
   if processing_details:
       successful_details = [d for d in processing_details if d['success']]
       if successful_details:
           processing_times = [d['processing_time'] for d in successful_details]
           fastest_time = min(processing_times)
           slowest_time = max(processing_times)
           
           log_master_message(f"  ‚ö° Performance Insights:")
           log_master_message(f"    Fastest directory: {fastest_time:.1f}s")
           log_master_message(f"    Slowest directory: {slowest_time:.1f}s")
           
           # Find fastest and slowest directories
           fastest_dir = min(successful_details, key=lambda x: x['processing_time'])
           slowest_dir = max(successful_details, key=lambda x: x['processing_time'])
           log_master_message(f"    Fastest: {fastest_dir['directory_identifier']}")
           log_master_message(f"    Slowest: {slowest_dir['directory_identifier']}")
   
   # Error analysis
   if failed_directories > 0:
       log_master_message(f"")
       log_master_message(f"‚ö†Ô∏è Error Analysis:")
       log_master_message(f"  Failed directories: {failed_directories}")
       
       failed_details = [d for d in processing_details if not d['success']]
       if failed_details:
           log_master_message(f"  Failed directory details:")
           for detail in failed_details:
               dir_identifier = detail['directory_identifier']
               error_info = detail.get('error_message', 'Unknown error')
               log_master_message(f"    ‚Ä¢ {dir_identifier}: {error_info}")
       
       log_master_message(f"  Check master log for detailed error information")
   else:
       log_master_message(f"")
       log_master_message(f"Ì†ºÌæâ SUCCESS: All discovered directories processed successfully!")
   
   # Backup system information
   backup_dir = os.getenv('DOC_BACKUP_ABSOLUTE_PATH', 'Default (parent/doc_backups)')
   log_master_message(f"")
   log_master_message(f"Ì†ΩÌ≤æ Backup System:")
   log_master_message(f"  Backup directory: {backup_dir}")
   log_master_message(f"  All subdirectories use the same backup location")
   log_master_message(f"  Preserves original directory structure")
   
   log_master_message(f"")
   log_master_message(f"Ì†ΩÌ≥ã Logs and Reports:")
   log_master_message(f"  Master processing log: ./logs/master_indexer.log")
   log_master_message(f"  Individual directory logs: ./logs/ (per-directory)")
   log_master_message(f"  Backup location: {backup_dir}")
   log_master_message(f"{'='*60}")


def main():
   """
   Dynamic main function - discovers and processes Year/Number directory structure
   """
   print("Dynamic Two-Level Master RAG Document Indexer Controller")
   print("=" * 60)
   print("Ì†ΩÌ¥ç DYNAMIC DISCOVERY APPROACH:")
   print("  ‚Ä¢ Dynamically discovers ANY year directories (2015, 2018, 2025, etc.)")
   print("  ‚Ä¢ Finds numbered subdirectories within each year")
   print("  ‚Ä¢ Excludes service directories (doc_backups, logs, etc.)")
   print("  ‚Ä¢ Calls indexer.py for each numbered subdirectory individually")
   print("  ‚Ä¢ Processes: Year/Number structure whatever actually exists")
   print("=" * 60)
   
   # Load environment variables
   load_dotenv()
   
   # Get root directory from environment or use default
   root_directory = os.getenv("MASTER_DOCUMENTS_DIR", "./data/634")
   
   # Set backup directory for all subdirectories if not configured
   if not os.getenv("DOC_BACKUP_ABSOLUTE_PATH"):
       master_backup_dir = os.path.join(os.path.dirname(root_directory), "doc_backups")
       os.environ['DOC_BACKUP_ABSOLUTE_PATH'] = master_backup_dir
       print(f"Ì†ΩÌ≤æ Backup directory: {master_backup_dir}")
   else:
       print(f"Ì†ΩÌ≤æ Using configured backup directory: {os.getenv('DOC_BACKUP_ABSOLUTE_PATH')}")
   
   log_master_message(f"Dynamic Master Indexer started")
   log_master_message(f"Root directory: {root_directory}")
   log_master_message(f"Backup directory: {os.getenv('DOC_BACKUP_ABSOLUTE_PATH', 'Not set')}")
   log_master_message(f"Service directories excluded: {', '.join(EXCLUDED_DIRECTORIES)}")
   
   # Dynamically discover all processing directories
   all_processing_directories = discover_all_processing_directories(root_directory)
   
   if not all_processing_directories:
       log_master_message(f"ERROR: No valid numbered subdirectories found in {root_directory}")
       print("ERROR: No valid directories found to process")
       print("No Year/Number directory structure found or all directories were excluded")
       sys.exit(1)
   
   # Initialize counters
   total_directories = len(all_processing_directories)
   successful_directories = 0
   failed_directories = 0
   processing_details = []
   
   # Count unique years for summary
   year_directories_found = len(set(Path(d).parent for d in all_processing_directories))
   
   log_master_message(f"Starting dynamic processing of {total_directories} numbered directories")
   log_master_message(f"Spanning {year_directories_found} year directories")
   
   master_start_time = time.time()
   
   # Process each discovered directory
   for index, directory_path in enumerate(all_processing_directories, 1):
       # Create readable directory identifier
       parts = Path(directory_path).parts
       if len(parts) >= 2:
           directory_identifier = f"{parts[-2]}/{parts[-1]}"
       else:
           directory_identifier = os.path.basename(directory_path)
       
       try:
           # Process directory using indexer.py
           success, processing_time, error_message = process_single_directory(
               directory_path, index, total_directories
           )
           
           # Record processing details
           detail = {
               'directory_identifier': directory_identifier,
               'directory_path': directory_path,
               'success': success,
               'processing_time': processing_time,
               'error_message': error_message
           }
           processing_details.append(detail)
           
           if success:
               successful_directories += 1
               log_master_message(f"‚úÖ Directory {directory_identifier} completed successfully")
           else:
               failed_directories += 1
               log_master_message(f"‚ùå Directory {directory_identifier} failed: {error_message}")
           
           # Progress update
           if index < total_directories:
               remaining = total_directories - index
               log_master_message(f"Progress: {index}/{total_directories} complete, {remaining} remaining")
           
       except KeyboardInterrupt:
           log_master_message(f"INTERRUPTED: Dynamic master indexer interrupted by user")
           log_master_message(f"Processed {successful_directories} directories successfully before interruption")
           print("\nDynamic master indexer interrupted by user")
           
           # Create partial summary
           total_time = time.time() - master_start_time
           create_final_summary(
               index, successful_directories, failed_directories, 
               total_time, processing_details, year_directories_found
           )
           sys.exit(1)
       
       except Exception as e:
           failed_directories += 1
           log_master_message(f"FATAL ERROR processing {directory_identifier}: {e}")
           
           # Add error details
           error_detail = {
               'directory_identifier': directory_identifier,
               'directory_path': directory_path,
               'success': False,
               'processing_time': 0,
               'error_message': str(e)
           }
           processing_details.append(error_detail)
   
   # Calculate total time
   total_time = time.time() - master_start_time
   
   # Create final summary
   create_final_summary(
       total_directories, successful_directories, failed_directories, 
       total_time, processing_details, year_directories_found
   )
   
   # Console summary
   print(f"\nÌ†ºÌæâ Dynamic Master Indexer Completed!")
   print(f"Ì†ΩÌ¥ç Discovery Results:")
   print(f"  Year directories found: {year_directories_found}")
   print(f"  Numbered directories found: {total_directories}")
   print(f"Ì†ΩÌ≥ä Processing Results:")
   print(f"  Successful: {successful_directories}")
   print(f"  Failed: {failed_directories}")
   print(f"  Total time: {total_time/60:.1f} minutes")
   
   print(f"Ì†ΩÌ≤æ Backup directory: {os.getenv('DOC_BACKUP_ABSOLUTE_PATH', 'Not configured')}")
   print(f"Ì†ΩÌ≥ã Master log: ./logs/master_indexer.log")
   
   # Exit with appropriate code
   if failed_directories > 0:
       print(f"‚ö†Ô∏è {failed_directories} directories failed - check logs for details")
       sys.exit(1)
   else:
       print(f"Ì†ºÌæâ All discovered directories processed successfully!")
       sys.exit(0)


if __name__ == "__main__":
   try:
       main()
   except KeyboardInterrupt:
       print("\nDynamic master indexer interrupted by user")
       log_master_message("Dynamic master indexer interrupted by user")
       sys.exit(1)
   except Exception as e:
       print(f"FATAL ERROR in dynamic master indexer: {e}")
       log_master_message(f"FATAL ERROR in dynamic master indexer: {e}")
       sys.exit(1)