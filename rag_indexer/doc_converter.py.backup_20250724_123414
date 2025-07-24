#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document converter module for RAG Document Indexer
Automatically converts .doc files to .docx format for proper processing
"""

import os
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import tempfile


class DocumentConverter:
    """Converter for .doc files to .docx format"""
    
    def __init__(self, backup_originals=True, delete_originals=False):
        """
        Initialize document converter
        
        Args:
            backup_originals: Whether to backup original .doc files
            delete_originals: Whether to delete .doc files after conversion
        """
        self.backup_originals = backup_originals
        self.delete_originals = delete_originals
        self.conversion_stats = {
            'attempted': 0,
            'successful': 0,
            'failed': 0,
            'failed_files': []
        }
        
        # Check if conversion tools are available
        self.libreoffice_available = self._check_libreoffice()
        self.pandoc_available = self._check_pandoc()
        
        if not self.libreoffice_available and not self.pandoc_available:
            print("WARNING: No document conversion tools found!")
            print("Install LibreOffice: sudo apt-get install libreoffice")
            print("Or install pandoc: sudo apt-get install pandoc")
    
    def _check_libreoffice(self):
        """Check if LibreOffice is available"""
        try:
            result = subprocess.run(['libreoffice', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def _check_pandoc(self):
        """Check if pandoc is available"""
        try:
            result = subprocess.run(['pandoc', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def _convert_with_libreoffice(self, doc_path, output_dir):
        """
        Convert .doc to .docx using LibreOffice
        
        Args:
            doc_path: Path to .doc file
            output_dir: Output directory for .docx file
        
        Returns:
            tuple: (success, docx_path, error_message)
        """
        try:
            cmd = [
                'libreoffice',
                '--headless',
                '--convert-to', 'docx',
                '--outdir', str(output_dir),
                str(doc_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Find the converted file
                doc_name = Path(doc_path).stem
                docx_path = Path(output_dir) / f"{doc_name}.docx"
                
                if docx_path.exists():
                    return True, docx_path, None
                else:
                    return False, None, "Converted file not found"
            else:
                return False, None, result.stderr or "LibreOffice conversion failed"
                
        except subprocess.TimeoutExpired:
            return False, None, "LibreOffice conversion timed out"
        except Exception as e:
            return False, None, f"LibreOffice error: {str(e)}"
    
    def _convert_with_pandoc(self, doc_path, output_path):
        """
        Convert .doc to .docx using pandoc
        
        Args:
            doc_path: Path to .doc file
            output_path: Path for output .docx file
        
        Returns:
            tuple: (success, docx_path, error_message)
        """
        try:
            cmd = ['pandoc', str(doc_path), '-o', str(output_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and Path(output_path).exists():
                return True, output_path, None
            else:
                return False, None, result.stderr or "Pandoc conversion failed"
                
        except subprocess.TimeoutExpired:
            return False, None, "Pandoc conversion timed out"
        except Exception as e:
            return False, None, f"Pandoc error: {str(e)}"
    
    def _backup_original_file(self, doc_path):
        """
        Create backup of original .doc file
        
        Args:
            doc_path: Path to original .doc file
        
        Returns:
            str: Path to backup file
        """
        try:
            backup_dir = Path(doc_path).parent / "doc_backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{Path(doc_path).stem}_backup_{timestamp}.doc"
            backup_path = backup_dir / backup_name
            
            shutil.copy2(doc_path, backup_path)
            return str(backup_path)
            
        except Exception as e:
            print(f"WARNING: Could not backup {doc_path}: {e}")
            return None
    
    def convert_single_file(self, doc_path, target_dir=None):
        """
        Convert a single .doc file to .docx
        
        Args:
            doc_path: Path to .doc file to convert
            target_dir: Directory to save .docx file (default: same as source)
        
        Returns:
            tuple: (success, docx_path, error_message)
        """
        doc_path = Path(doc_path)
        
        if not doc_path.exists():
            return False, None, f"File not found: {doc_path}"
        
        if doc_path.suffix.lower() != '.doc':
            return False, None, f"Not a .doc file: {doc_path}"
        
        # Determine target directory
        if target_dir is None:
            target_dir = doc_path.parent
        else:
            target_dir = Path(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
        
        # Target .docx file path
        docx_path = target_dir / f"{doc_path.stem}.docx"
        
        # Check if .docx already exists
        if docx_path.exists():
            print(f"INFO: {docx_path.name} already exists, skipping conversion")
            return True, docx_path, "Already exists"
        
        self.conversion_stats['attempted'] += 1
        
        # Backup original if requested
        backup_path = None
        if self.backup_originals:
            backup_path = self._backup_original_file(doc_path)
        
        # Try conversion with LibreOffice first
        success = False
        error_msg = None
        
        if self.libreoffice_available:
            print(f"Converting {doc_path.name} with LibreOffice...")
            success, result_path, error_msg = self._convert_with_libreoffice(doc_path, target_dir)
            
            if success:
                docx_path = result_path
        
        # Fallback to pandoc if LibreOffice failed
        if not success and self.pandoc_available:
            print(f"Retrying {doc_path.name} with pandoc...")
            success, result_path, error_msg = self._convert_with_pandoc(doc_path, docx_path)
            
            if success:
                docx_path = result_path
        
        # Update statistics
        if success:
            self.conversion_stats['successful'] += 1
            print(f"SUCCESS: Converted {doc_path.name} → {docx_path.name}")
            
            # Delete original if requested
            if self.delete_originals:
                try:
                    doc_path.unlink()
                    print(f"INFO: Deleted original {doc_path.name}")
                except Exception as e:
                    print(f"WARNING: Could not delete original {doc_path.name}: {e}")
            
            return True, docx_path, None
        else:
            self.conversion_stats['failed'] += 1
            self.conversion_stats['failed_files'].append(str(doc_path))
            print(f"ERROR: Failed to convert {doc_path.name}: {error_msg}")
            return False, None, error_msg
    
    def scan_and_convert_directory(self, directory_path, recursive=True):
        """
        Scan directory for .doc files and convert them to .docx
        
        Args:
            directory_path: Directory to scan
            recursive: Whether to scan subdirectories
        
        Returns:
            dict: Conversion results
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            return {'error': f"Directory not found: {directory_path}"}
        
        print(f"Scanning for .doc files in: {directory_path}")
        
        # Find all .doc files
        if recursive:
            doc_files = list(directory_path.rglob("*.doc"))
        else:
            doc_files = list(directory_path.glob("*.doc"))
        
        if not doc_files:
            print("INFO: No .doc files found")
            return {
                'attempted': 0,
                'successful': 0,
                'failed': 0,
                'failed_files': [],
                'message': 'No .doc files found'
            }
        
        print(f"Found {len(doc_files)} .doc files to convert")
        
        # Convert each file
        converted_files = []
        for doc_file in doc_files:
            success, docx_path, error = self.convert_single_file(doc_file)
            if success:
                converted_files.append(str(docx_path))
        
        # Return results
        return {
            'attempted': self.conversion_stats['attempted'],
            'successful': self.conversion_stats['successful'],
            'failed': self.conversion_stats['failed'],
            'failed_files': self.conversion_stats['failed_files'],
            'converted_files': converted_files,
            'success_rate': (self.conversion_stats['successful'] / self.conversion_stats['attempted'] * 100) if self.conversion_stats['attempted'] > 0 else 0
        }
    
    def print_conversion_summary(self):
        """Print summary of conversion operations"""
        print("\nDocument Conversion Summary:")
        print("=" * 40)
        print(f"Files attempted: {self.conversion_stats['attempted']}")
        print(f"Successfully converted: {self.conversion_stats['successful']}")
        print(f"Failed conversions: {self.conversion_stats['failed']}")
        
        if self.conversion_stats['attempted'] > 0:
            success_rate = (self.conversion_stats['successful'] / self.conversion_stats['attempted']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        if self.conversion_stats['failed_files']:
            print(f"\nFailed files:")
            for failed_file in self.conversion_stats['failed_files']:
                print(f"  - {failed_file}")
        
        print("=" * 40)


def convert_doc_files_in_directory(directory_path, recursive=True, backup_originals=True, delete_originals=False):
    """
    Convenience function to convert all .doc files in a directory
    
    Args:
        directory_path: Directory to scan and convert
        recursive: Whether to scan subdirectories
        backup_originals: Whether to backup original files
        delete_originals: Whether to delete original files after conversion
    
    Returns:
        dict: Conversion results
    """
    converter = DocumentConverter(
        backup_originals=backup_originals,
        delete_originals=delete_originals
    )
    
    results = converter.scan_and_convert_directory(directory_path, recursive)
    converter.print_conversion_summary()
    
    return results


def check_conversion_tools():
    """
    Check which document conversion tools are available
    
    Returns:
        dict: Available tools information
    """
    converter = DocumentConverter()
    
    tools_info = {
        'libreoffice_available': converter.libreoffice_available,
        'pandoc_available': converter.pandoc_available,
        'any_tool_available': converter.libreoffice_available or converter.pandoc_available
    }
    
    print("Document Conversion Tools Status:")
    print("=" * 40)
    print(f"LibreOffice: {'✅ Available' if tools_info['libreoffice_available'] else '❌ Not available'}")
    print(f"Pandoc: {'✅ Available' if tools_info['pandoc_available'] else '❌ Not available'}")
    
    if not tools_info['any_tool_available']:
        print("\n⚠️ WARNING: No conversion tools available!")
        print("Install with:")
        print("  sudo apt-get install libreoffice")
        print("  sudo apt-get install pandoc")
    
    print("=" * 40)
    
    return tools_info


if __name__ == "__main__":
    # Example usage
    print("Document Converter Test")
    check_conversion_tools()
    
    # Test conversion in current directory
    # results = convert_doc_files_in_directory("./data", recursive=True)
    # print(f"Conversion completed: {results}")
