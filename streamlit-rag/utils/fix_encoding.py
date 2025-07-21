# utils/fix_encoding.py
# Utility to fix encoding issues in Python files

import os
import chardet
import codecs
from pathlib import Path

def detect_encoding(file_path):
    """Detect file encoding"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding']
    except Exception as e:
        print(f"Error detecting encoding for {file_path}: {e}")
        return None

def fix_file_encoding(file_path, target_encoding='utf-8'):
    """Fix file encoding by converting to UTF-8"""
    try:
        # Detect current encoding
        current_encoding = detect_encoding(file_path)
        if not current_encoding:
            print(f"Could not detect encoding for {file_path}")
            return False
        
        print(f"Converting {file_path}: {current_encoding} -> {target_encoding}")
        
        # Read with detected encoding
        with codecs.open(file_path, 'r', encoding=current_encoding, errors='ignore') as f:
            content = f.read()
        
        # Write with UTF-8 encoding
        with codecs.open(file_path, 'w', encoding=target_encoding) as f:
            f.write(content)
        
        print(f"? Successfully converted {file_path}")
        return True
        
    except Exception as e:
        print(f"? Error fixing {file_path}: {e}")
        return False

def fix_project_encoding(project_dir="."):
    """Fix encoding for all Python files in project"""
    print("?? Fixing encoding issues in project files...")
    
    python_files = list(Path(project_dir).rglob("*.py"))
    
    for file_path in python_files:
        if fix_file_encoding(str(file_path)):
            print(f"  ? Fixed: {file_path}")
        else:
            print(f"  ?? Skipped: {file_path}")
    
    print(f"\n?? Processed {len(python_files)} Python files")

if __name__ == "__main__":
    fix_project_encoding()