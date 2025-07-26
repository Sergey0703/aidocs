#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for PDF processor module
Tests PDF extraction capabilities and chunking behavior
"""

import os
import sys
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

# Import our PDF processor
try:
    from pdf_processor import create_pdf_processor, check_pdf_processing_capabilities
    from file_utils_core import scan_files_in_directory
    print("? PDF processor imports successful")
except ImportError as e:
    print(f"? Import error: {e}")
    print("Make sure pdf_processor.py and file_utils_core.py are in the same directory")
    sys.exit(1)

# Import for chunking simulation
try:
    from llama_index.core.node_parser import SentenceSplitter
    CHUNKING_AVAILABLE = True
except ImportError:
    CHUNKING_AVAILABLE = False
    print("?? LlamaIndex not available - chunking simulation disabled")


class PDFProcessorTester:
    """Tester for PDF processing capabilities"""
    
    def __init__(self, test_directory):
        """
        Initialize tester
        
        Args:
            test_directory: Directory containing PDF files to test
        """
        self.test_directory = test_directory
        self.pdf_processor = None
        
        # Chunking settings (from your .env)
        self.chunk_size = 2048
        self.chunk_overlap = 256
        self.min_chunk_length = 100
        
        # Initialize node parser for chunking simulation
        if CHUNKING_AVAILABLE:
            self.node_parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                include_metadata=True
            )
        else:
            self.node_parser = None
    
    def run_capability_check(self):
        """Check PDF processing capabilities"""
        print("=" * 60)
        print("?? CHECKING PDF PROCESSING CAPABILITIES")
        print("=" * 60)
        
        capabilities = check_pdf_processing_capabilities()
        
        if capabilities['overall_status'] == 'ready':
            print("? PDF processing ready!")
        else:
            print("?? Limited PDF processing capabilities")
            print("Install missing libraries: pip install PyMuPDF pdfplumber pdf2image")
        
        return capabilities
    
    def find_test_pdfs(self):
        """Find PDF files in test directory"""
        print(f"\n?? Scanning for PDF files in: {self.test_directory}")
        
        if not os.path.exists(self.test_directory):
            print(f"? Directory does not exist: {self.test_directory}")
            return []
        
        # Find all PDF files
        all_files = scan_files_in_directory(self.test_directory, recursive=True)
        pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
        
        print(f"?? Found {len(pdf_files)} PDF files")
        
        # Show first few for reference
        if pdf_files:
            print("First few PDF files:")
            for i, pdf_file in enumerate(pdf_files[:5], 1):
                file_size = os.path.getsize(pdf_file) / 1024  # KB
                print(f"  {i}. {os.path.basename(pdf_file)} ({file_size:.1f} KB)")
            
            if len(pdf_files) > 5:
                print(f"  ... and {len(pdf_files) - 5} more")
        
        return pdf_files
    
    def simulate_chunking(self, document):
        """
        Simulate how the document would be chunked
        
        Args:
            document: Document object
        
        Returns:
            dict: Chunking analysis
        """
        if not self.node_parser:
            return {'error': 'Chunking not available'}
        
        try:
            # Create nodes (chunks) from document
            nodes = self.node_parser.get_nodes_from_documents([document])
            
            # Analyze chunks
            chunk_analysis = {
                'total_chunks': len(nodes),
                'chunk_sizes': [len(node.get_content()) for node in nodes],
                'total_characters': sum(len(node.get_content()) for node in nodes),
                'average_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
            
            if chunk_analysis['chunk_sizes']:
                chunk_analysis['average_chunk_size'] = sum(chunk_analysis['chunk_sizes']) / len(chunk_analysis['chunk_sizes'])
                chunk_analysis['min_chunk_size'] = min(chunk_analysis['chunk_sizes'])
                chunk_analysis['max_chunk_size'] = max(chunk_analysis['chunk_sizes'])
            
            return chunk_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def test_single_pdf(self, pdf_path, detailed=False):
        """
        Test processing of a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            detailed: Whether to show detailed output
        
        Returns:
            dict: Test results
        """
        print(f"\n?? Testing: {os.path.basename(pdf_path)}")
        
        if not self.pdf_processor:
            self.pdf_processor = create_pdf_processor()
        
        start_time = time.time()
        
        try:
            # Process PDF
            documents = self.pdf_processor.process_pdf_file(pdf_path)
            
            processing_time = time.time() - start_time
            
            if not documents:
                return {
                    'success': False,
                    'error': 'No documents created',
                    'processing_time': processing_time
                }
            
            document = documents[0]  # Should be one document per PDF
            
            # Analyze content
            content = document.text
            content_length = len(content)
            word_count = len(content.split())
            
            # Simulate chunking
            chunk_analysis = self.simulate_chunking(document)
            
            # Create result
            result = {
                'success': True,
                'processing_time': processing_time,
                'content_length': content_length,
                'word_count': word_count,
                'chunk_analysis': chunk_analysis,
                'extraction_method': document.metadata.get('extraction_info', {}).get('method', 'unknown'),
                'pdf_type': document.metadata.get('pdf_analysis', {}).get('type', 'unknown')
            }
            
            # Print summary
            print(f"   ? SUCCESS: {content_length:,} chars, {word_count:,} words in {processing_time:.2f}s")
            print(f"   ?? Method: {result['extraction_method']}, Type: {result['pdf_type']}")
            
            if 'error' not in chunk_analysis:
                chunks = chunk_analysis['total_chunks']
                avg_size = chunk_analysis['average_chunk_size']
                print(f"   ?? Chunking: {chunks} chunks, avg {avg_size:.0f} chars/chunk")
                
                # Chunking ratio analysis
                if chunks == 1:
                    print(f"   ?? Result: 1 PDF ? 1 chunk (small file)")
                else:
                    print(f"   ?? Result: 1 PDF ? {chunks} chunks (will be split)")
            
            # Show content preview if detailed
            if detailed:
                print(f"\n?? Content preview (first 200 chars):")
                preview = content[:200].replace('\n', ' ')
                print(f"   \"{preview}...\"")
                
                if 'error' not in chunk_analysis:
                    print(f"\n?? Chunk size distribution:")
                    sizes = chunk_analysis['chunk_sizes']
                    print(f"   Min: {chunk_analysis['min_chunk_size']} chars")
                    print(f"   Max: {chunk_analysis['max_chunk_size']} chars")
                    print(f"   Avg: {chunk_analysis['average_chunk_size']:.0f} chars")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"   ? ERROR: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }
    
    def run_batch_test(self, max_files=5):
        """
        Test multiple PDF files
        
        Args:
            max_files: Maximum number of files to test
        
        Returns:
            dict: Batch test results
        """
        print("=" * 60)
        print("?? RUNNING BATCH PDF TESTS")
        print("=" * 60)
        
        pdf_files = self.find_test_pdfs()
        
        if not pdf_files:
            print("? No PDF files found for testing")
            return {'error': 'No PDF files found'}
        
        # Limit number of files to test
        test_files = pdf_files[:max_files]
        
        print(f"\n?? Testing {len(test_files)} PDF files...")
        
        results = []
        total_start_time = time.time()
        
        for i, pdf_path in enumerate(test_files, 1):
            print(f"\n--- Test {i}/{len(test_files)} ---")
            result = self.test_single_pdf(pdf_path, detailed=(i <= 2))  # Detailed for first 2
            result['file_name'] = os.path.basename(pdf_path)
            results.append(result)
        
        total_time = time.time() - total_start_time
        
        # Analyze batch results
        successful_tests = [r for r in results if r['success']]
        failed_tests = [r for r in results if not r['success']]
        
        print("\n" + "=" * 60)
        print("?? BATCH TEST SUMMARY")
        print("=" * 60)
        
        print(f"?? Files tested: {len(results)}")
        print(f"? Successful: {len(successful_tests)}")
        print(f"? Failed: {len(failed_tests)}")
        print(f"?? Total time: {total_time:.2f}s")
        
        if successful_tests:
            print(f"\n?? SUCCESS STATISTICS:")
            
            # Content statistics
            total_chars = sum(r['content_length'] for r in successful_tests)
            total_words = sum(r['word_count'] for r in successful_tests)
            avg_chars = total_chars / len(successful_tests)
            avg_words = total_words / len(successful_tests)
            
            print(f"   Total content: {total_chars:,} characters, {total_words:,} words")
            print(f"   Average per file: {avg_chars:,.0f} chars, {avg_words:,.0f} words")
            
            # Chunking statistics
            chunk_counts = []
            for r in successful_tests:
                if 'error' not in r['chunk_analysis']:
                    chunk_counts.append(r['chunk_analysis']['total_chunks'])
            
            if chunk_counts:
                total_chunks = sum(chunk_counts)
                avg_chunks = total_chunks / len(chunk_counts)
                
                print(f"\n?? CHUNKING STATISTICS:")
                print(f"   Total chunks: {total_chunks}")
                print(f"   Average chunks per PDF: {avg_chunks:.1f}")
                print(f"   Files ? Chunks ratio: 1 PDF ? {avg_chunks:.1f} chunks")
                
                # Distribution analysis
                single_chunk_files = sum(1 for c in chunk_counts if c == 1)
                multi_chunk_files = len(chunk_counts) - single_chunk_files
                
                print(f"   Single chunk files: {single_chunk_files}/{len(chunk_counts)}")
                print(f"   Multi-chunk files: {multi_chunk_files}/{len(chunk_counts)}")
            
            # Processing speed
            avg_processing_time = sum(r['processing_time'] for r in successful_tests) / len(successful_tests)
            print(f"\n? PERFORMANCE:")
            print(f"   Average processing time: {avg_processing_time:.2f}s per file")
            print(f"   Speed: {avg_chars/avg_processing_time:.0f} chars/second")
            
            # Method distribution
            methods = {}
            pdf_types = {}
            for r in successful_tests:
                method = r['extraction_method']
                pdf_type = r['pdf_type']
                methods[method] = methods.get(method, 0) + 1
                pdf_types[pdf_type] = pdf_types.get(pdf_type, 0) + 1
            
            print(f"\n?? EXTRACTION METHODS USED:")
            for method, count in methods.items():
                print(f"   {method}: {count} files")
            
            print(f"\n?? PDF TYPES DETECTED:")
            for pdf_type, count in pdf_types.items():
                print(f"   {pdf_type}: {count} files")
        
        if failed_tests:
            print(f"\n? FAILED FILES:")
            for r in failed_tests:
                print(f"   {r['file_name']}: {r['error']}")
        
        # Get processor statistics
        if self.pdf_processor:
            print(f"\n?? PROCESSOR STATISTICS:")
            self.pdf_processor.print_processing_summary()
        
        return {
            'total_files': len(results),
            'successful': len(successful_tests),
            'failed': len(failed_tests),
            'total_time': total_time,
            'results': results
        }
    
    def run_complete_analysis(self):
        """
        Analyze ALL PDF files and find top 5 with most chunks
        
        Returns:
            dict: Complete analysis results
        """
        print("=" * 60)
        print("?? RUNNING COMPLETE PDF ANALYSIS")
        print("=" * 60)
        
        pdf_files = self.find_test_pdfs()
        
        if not pdf_files:
            print("? No PDF files found for analysis")
            return {'error': 'No PDF files found'}
        
        print(f"\n?? Analyzing ALL {len(pdf_files)} PDF files...")
        print("? This may take a while for large collections...")
        
        all_results = []
        total_start_time = time.time()
        
        # Process all files with progress indicator
        for i, pdf_path in enumerate(pdf_files, 1):
            if i % 10 == 0 or i == len(pdf_files):  # Progress every 10 files
                print(f"?? Progress: {i}/{len(pdf_files)} files analyzed...")
            
            result = self.test_single_pdf(pdf_path, detailed=False)  # No detailed output for all
            result['file_name'] = os.path.basename(pdf_path)
            result['file_path'] = pdf_path
            all_results.append(result)
        
        total_time = time.time() - total_start_time
        
        # Filter successful results
        successful_results = [r for r in all_results if r['success']]
        failed_results = [r for r in all_results if not r['success']]
        
        print(f"\n? Analysis complete! Processed {len(all_results)} files in {total_time:.2f}s")
        print(f"   Successful: {len(successful_results)}")
        print(f"   Failed: {len(failed_results)}")
        
        if not successful_results:
            print("? No successful extractions found")
            return {'error': 'No successful extractions'}
        
        # Find files with chunk information
        files_with_chunks = []
        for result in successful_results:
            if 'error' not in result['chunk_analysis']:
                chunk_count = result['chunk_analysis']['total_chunks']
                files_with_chunks.append({
                    'file_name': result['file_name'],
                    'file_path': result['file_path'],
                    'chunk_count': chunk_count,
                    'content_length': result['content_length'],
                    'word_count': result['word_count'],
                    'processing_time': result['processing_time'],
                    'extraction_method': result['extraction_method'],
                    'pdf_type': result['pdf_type'],
                    'avg_chunk_size': result['chunk_analysis']['average_chunk_size']
                })
        
        if not files_with_chunks:
            print("? No files with chunk analysis found")
            return {'error': 'No chunk data available'}
        
        # Sort by chunk count (descending) and get top 5
        files_with_chunks.sort(key=lambda x: x['chunk_count'], reverse=True)
        top_5_files = files_with_chunks[:5]
        
        print("\n" + "=" * 60)
        print("?? TOP 5 FILES WITH MOST CHUNKS")
        print("=" * 60)
        
        for i, file_info in enumerate(top_5_files, 1):
            print(f"\n?? #{i} - {file_info['file_name']}")
            print(f"   ?? Chunks: {file_info['chunk_count']}")
            print(f"   ?? Content: {file_info['content_length']:,} chars, {file_info['word_count']:,} words")
            print(f"   ?? Avg chunk size: {file_info['avg_chunk_size']:.0f} chars")
            print(f"   ?? Method: {file_info['extraction_method']}")
            print(f"   ?? Processing: {file_info['processing_time']:.2f}s")
        
        # Overall statistics
        print("\n" + "=" * 60)
        print("?? COMPLETE ANALYSIS STATISTICS")
        print("=" * 60)
        
        print(f"?? Total files: {len(all_results)}")
        print(f"? Successfully processed: {len(successful_results)}")
        print(f"? Failed: {len(failed_results)}")
        print(f"?? Total processing time: {total_time:.2f}s")
        
        if successful_results:
            # Content statistics
            total_chars = sum(r['content_length'] for r in successful_results)
            total_words = sum(r['word_count'] for r in successful_results)
            avg_chars = total_chars / len(successful_results)
            avg_words = total_words / len(successful_results)
            
            print(f"\n?? CONTENT STATISTICS:")
            print(f"   Total content: {total_chars:,} characters, {total_words:,} words")
            print(f"   Average per file: {avg_chars:,.0f} chars, {avg_words:,.0f} words")
            
            # Chunking statistics
            if files_with_chunks:
                chunk_counts = [f['chunk_count'] for f in files_with_chunks]
                total_chunks = sum(chunk_counts)
                avg_chunks = total_chunks / len(chunk_counts)
                max_chunks = max(chunk_counts)
                min_chunks = min(chunk_counts)
                
                print(f"\n?? CHUNKING STATISTICS:")
                print(f"   Total chunks: {total_chunks}")
                print(f"   Average chunks per PDF: {avg_chunks:.2f}")
                print(f"   Min chunks: {min_chunks}")
                print(f"   Max chunks: {max_chunks}")
                
                # Distribution analysis
                single_chunk_files = sum(1 for c in chunk_counts if c == 1)
                multi_chunk_files = len(chunk_counts) - single_chunk_files
                big_files = sum(1 for c in chunk_counts if c >= 5)  # 5+ chunks
                
                print(f"   Single chunk files: {single_chunk_files}/{len(chunk_counts)} ({single_chunk_files/len(chunk_counts)*100:.1f}%)")
                print(f"   Multi-chunk files: {multi_chunk_files}/{len(chunk_counts)} ({multi_chunk_files/len(chunk_counts)*100:.1f}%)")
                print(f"   Large files (5+ chunks): {big_files}/{len(chunk_counts)} ({big_files/len(chunk_counts)*100:.1f}%)")
            
            # Performance statistics
            avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
            print(f"\n? PERFORMANCE STATISTICS:")
            print(f"   Average processing time: {avg_processing_time:.3f}s per file")
            print(f"   Processing speed: {avg_chars/avg_processing_time:.0f} chars/second")
            print(f"   Files per minute: {60/avg_processing_time:.0f} files/min")
            
            # Method and type distribution
            methods = {}
            pdf_types = {}
            for r in successful_results:
                method = r['extraction_method']
                pdf_type = r['pdf_type']
                methods[method] = methods.get(method, 0) + 1
                pdf_types[pdf_type] = pdf_types.get(pdf_type, 0) + 1
            
            print(f"\n?? EXTRACTION METHODS:")
            for method, count in methods.items():
                percentage = count / len(successful_results) * 100
                print(f"   {method}: {count} files ({percentage:.1f}%)")
            
            print(f"\n?? PDF TYPES:")
            for pdf_type, count in pdf_types.items():
                percentage = count / len(successful_results) * 100
                print(f"   {pdf_type}: {count} files ({percentage:.1f}%)")
        
        if failed_results:
            print(f"\n? FAILED FILES (first 10):")
            for r in failed_results[:10]:
                print(f"   {r['file_name']}: {r['error']}")
            if len(failed_results) > 10:
                print(f"   ... and {len(failed_results) - 10} more failed files")
        
        # Detailed analysis of top files
        if top_5_files:
            print(f"\n?? DETAILED ANALYSIS OF TOP FILES:")
            for i, file_info in enumerate(top_5_files, 1):
                if file_info['chunk_count'] > 1:  # Only show multi-chunk files
                    print(f"\n?? #{i} {file_info['file_name']}")
                    print(f"   ?? Will create {file_info['chunk_count']} searchable chunks")
                    print(f"   ?? Total content: {file_info['content_length']:,} characters")
                    print(f"   ?? Average chunk: {file_info['avg_chunk_size']:.0f} characters")
                    print(f"   ?? Good for: Detailed content search and retrieval")
        
        # Save detailed results for further analysis
        analysis_results = {
            'total_files': len(all_results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'total_time': total_time,
            'top_5_files': top_5_files,
            'all_results': all_results,
            'chunk_statistics': {
                'total_chunks': sum(f['chunk_count'] for f in files_with_chunks) if files_with_chunks else 0,
                'avg_chunks_per_file': sum(f['chunk_count'] for f in files_with_chunks) / len(files_with_chunks) if files_with_chunks else 0,
                'files_with_multiple_chunks': sum(1 for f in files_with_chunks if f['chunk_count'] > 1) if files_with_chunks else 0
            }
        }
        
        return analysis_results


    def run_quick_test(self):
        """Run a quick test on first available PDF"""
        print("=" * 60)
        print("? QUICK PDF TEST")
        print("=" * 60)
        
        pdf_files = self.find_test_pdfs()
        
        if not pdf_files:
            print("? No PDF files found for testing")
            return False
        
        # Test just the first file
        first_pdf = pdf_files[0]
        result = self.test_single_pdf(first_pdf, detailed=True)
        
        return result['success'] if result else False


def main():
    """Main test function"""
    print("?? PDF PROCESSOR TESTING SUITE")
    print("=" * 60)
    
    # Get test directory from environment or use default
    test_directory = os.getenv("DOCUMENTS_DIR", "./data/634/2025/1")
    print(f"?? Test directory: {test_directory}")
    
    # Create tester
    tester = PDFProcessorTester(test_directory)
    
    # Run capability check
    capabilities = tester.run_capability_check()
    
    if capabilities['overall_status'] != 'ready':
        print("\n?? PDF processing capabilities limited. Install missing libraries first.")
        return
    
    # Ask user what test to run
    print("\n?? TEST OPTIONS:")
    print("1. Quick test (1 PDF file)")
    print("2. Batch test (up to 5 PDF files)")
    print("3. Standard analysis (up to 10 PDF files)")
    print("4. Complete analysis (ALL files + top 5 with most chunks)")
    
    try:
        choice = input("\nChoose test (1-4) [default: 1]: ").strip()
        if not choice:
            choice = "1"
        
        if choice == "1":
            success = tester.run_quick_test()
            if success:
                print("\n? Quick test PASSED")
            else:
                print("\n? Quick test FAILED")
        
        elif choice == "2":
            results = tester.run_batch_test(max_files=5)
            if results and results.get('successful', 0) > 0:
                print("\n? Batch test completed with some successes")
            else:
                print("\n? Batch test failed")
        
        elif choice == "3":
            results = tester.run_batch_test(max_files=10)
            print("\n?? Standard analysis completed")
        
        elif choice == "4":
            print("\n?? Starting complete analysis of ALL PDF files...")
            results = tester.run_complete_analysis()
            if results and not results.get('error'):
                print("\n? Complete analysis finished successfully!")
                if results.get('top_5_files'):
                    print(f"?? Found {len(results['top_5_files'])} files with detailed chunk information")
                    print(f"?? Total chunks across all files: {results['chunk_statistics']['total_chunks']}")
                    multi_chunk_files = results['chunk_statistics']['files_with_multiple_chunks']
                    if multi_chunk_files > 0:
                        print(f"?? Files that will create multiple chunks: {multi_chunk_files}")
                        print("?? These files contain substantial content for search and retrieval!")
                    else:
                        print("?? Most files appear to be certificates or short documents")
            else:
                print("\n? Complete analysis failed")
        
        else:
            print("? Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\n?? Test interrupted by user")
    except Exception as e:
        print(f"\n? Test error: {e}")


if __name__ == "__main__":
    main()