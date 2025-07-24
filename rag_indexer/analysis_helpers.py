#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis helpers module for RAG Document Indexer
Contains helper functions for result analysis and reporting
"""

from datetime import datetime
from utils import save_failed_files_details


def analyze_final_results_enhanced(config, db_manager, log_dir, processing_stats):
    """
    Enhanced end-to-end analysis with comprehensive reporting
    
    Args:
        config: Configuration object
        db_manager: Database manager instance
        log_dir: Directory for log files
        processing_stats: Processing statistics from all stages
    
    Returns:
        dict: Comprehensive analysis results
    """
    print(f"\n{'='*70}")
    print("🔍 ENHANCED END-TO-END ANALYSIS")
    print(f"{'='*70}")
    
    # Perform comprehensive directory vs database comparison
    analysis_results = db_manager.analyze_directory_vs_database(
        config.DOCUMENTS_DIR, 
        recursive=True
    )
    
    # Extract results
    total_files = analysis_results['total_files_in_directory']
    files_in_db = analysis_results['files_successfully_in_db']
    missing_files = analysis_results['files_missing_from_db']
    missing_files_detailed = analysis_results['missing_files_detailed']
    success_rate = analysis_results['success_rate']
    
    print(f"\n📊 Final Processing Results:")
    print(f"   📁 Total files in directory: {total_files:,}")
    print(f"   ✅ Files successfully in database: {files_in_db:,}")
    print(f"   ❌ Files missing from database: {missing_files:,}")
    print(f"   📈 End-to-end success rate: {success_rate:.1f}%")
    
    # Enhanced analysis with processing stages
    if processing_stats:
        print_pipeline_analysis(processing_stats)
    
    # Handle failed files analysis
    failure_analysis = analyze_failed_files(missing_files_detailed, log_dir)
    
    # Enhanced results with processing context
    enhanced_analysis = {
        **analysis_results,  # Include original analysis
        'processing_stats': processing_stats,
        'pipeline_success': processing_stats.get('records_saved', 0) > 0,
        'feature_effectiveness': analyze_feature_effectiveness(processing_stats),
        'performance_metrics': calculate_performance_metrics(processing_stats),
        'failure_analysis': failure_analysis
    }
    
    return enhanced_analysis


def print_pipeline_analysis(processing_stats):
    """
    Print detailed pipeline analysis
    
    Args:
        processing_stats: Processing statistics dictionary
    """
    print(f"\n🔄 Processing Pipeline Analysis:")
    
    # Document loading stage
    if 'documents_loaded' in processing_stats:
        print(f"   📄 Documents loaded: {processing_stats['documents_loaded']:,}")
    
    if 'images_processed' in processing_stats:
        print(f"   🖼️ Images processed: {processing_stats['images_processed']:,}")
    
    # Chunk processing stage  
    if 'chunks_created' in processing_stats:
        print(f"   🧩 Chunks created: {processing_stats['chunks_created']:,}")
    
    if 'valid_chunks' in processing_stats:
        print(f"   ✅ Valid chunks: {processing_stats['valid_chunks']:,}")
    
    # Database stage
    if 'records_saved' in processing_stats:
        print(f"   💾 Records saved to database: {processing_stats['records_saved']:,}")
    
    # Calculate pipeline loss at each stage
    total_attempted = processing_stats.get('total_nodes', processing_stats.get('chunks_created', 0))
    if total_attempted > 0:
        saved_records = processing_stats.get('records_saved', 0)
        pipeline_loss = total_attempted - saved_records
        loss_rate = (pipeline_loss / total_attempted * 100)
        
        print(f"\n📉 Pipeline Loss Analysis:")
        print(f"   Total chunks attempted: {total_attempted:,}")
        print(f"   Successfully saved: {saved_records:,}")
        print(f"   Lost in pipeline: {pipeline_loss:,}")
        print(f"   Pipeline loss rate: {loss_rate:.2f}%")


def analyze_failed_files(missing_files_detailed, log_dir):
    """
    Analyze failed files and save detailed information
    
    Args:
        missing_files_detailed: List of detailed failure information
        log_dir: Directory for log files
    
    Returns:
        dict: Failure analysis results
    """
    if not missing_files_detailed:
        print(f"\n🎉 PERFECT PROCESSING: All files successfully indexed!")
        return {'total_failed': 0, 'categories': {}, 'perfect_processing': True}
    
    print(f"\n📝 Saving detailed analysis...")
    log_file_path = save_failed_files_details(missing_files_detailed, log_dir)
    
    if log_file_path:
        print(f"   📋 Missing files details saved to: {log_file_path}")
    else:
        print(f"   ⚠️ Could not save missing files details")
    
    # Show categorized failure analysis
    failure_categories = categorize_failures(missing_files_detailed)
    
    if failure_categories:
        print(f"\n🔍 Failure Category Analysis:")
        for category, count in sorted(failure_categories.items(), key=lambda x: x[1], reverse=True):
            print(f"   {category}: {count} files")
    
    # Show first few missing files for quick reference
    print(f"\n❌ Sample Missing Files:")
    for i, missing_detail in enumerate(missing_files_detailed[:5], 1):
        print(f"   {i}. {missing_detail}")
    if len(missing_files_detailed) > 5:
        print(f"   ... and {len(missing_files_detailed) - 5} more (see detailed log)")
    
    return {
        'total_failed': len(missing_files_detailed),
        'categories': failure_categories,
        'log_file': log_file_path,
        'sample_failures': missing_files_detailed[:10],  # First 10 for reference
        'perfect_processing': False
    }


def categorize_failures(missing_files_detailed):
    """
    Categorize failure reasons from detailed failure list
    
    Args:
        missing_files_detailed: List of detailed failure information
    
    Returns:
        dict: Dictionary of failure categories and counts
    """
    failure_categories = {}
    
    for missing_detail in missing_files_detailed:
        # Extract failure category from detail
        if " - " in missing_detail:
            category = missing_detail.split(" - ", 1)[1].split(" ")[0]
        else:
            category = "UNKNOWN"
        
        failure_categories[category] = failure_categories.get(category, 0) + 1
    
    return failure_categories


def analyze_feature_effectiveness(processing_stats):
    """
    Analyze effectiveness of enhanced features
    
    Args:
        processing_stats: Processing statistics dictionary
    
    Returns:
        dict: Feature effectiveness analysis
    """
    effectiveness = {}
    
    # OCR effectiveness
    if 'images_processed' in processing_stats:
        ocr_images = processing_stats['images_processed']
        if 'rotation_stats' in processing_stats and processing_stats['rotation_stats']:
            rotation_stats = processing_stats['rotation_stats']
            rotations_applied = rotation_stats.get('rotations_applied', 0)
            if ocr_images > 0:
                effectiveness['auto_rotation_usage'] = (rotations_applied / ocr_images * 100)
        
        if 'total_ocr_text_length' in processing_stats:
            avg_text_per_image = processing_stats['total_ocr_text_length'] / ocr_images if ocr_images > 0 else 0
            effectiveness['avg_ocr_text_extraction'] = avg_text_per_image
    
    # Advanced parsing effectiveness
    if 'method_usage' in processing_stats:
        method_usage = processing_stats['method_usage']
        advanced_used = method_usage.get('advanced_parsing', 0)
        total_files = advanced_used + method_usage.get('fallback_processing', 0)
        if total_files > 0:
            effectiveness['advanced_parsing_usage'] = (advanced_used / total_files * 100)
    
    # Quality filtering effectiveness
    if 'filter_success_rate' in processing_stats:
        effectiveness['quality_filter_success'] = processing_stats['filter_success_rate']
    
    # Document conversion effectiveness
    if 'conversion_results' in processing_stats:
        conversion = processing_stats['conversion_results']
        if conversion.get('attempted', 0) > 0:
            effectiveness['doc_conversion_success'] = (
                conversion.get('successful', 0) / conversion['attempted'] * 100
            )
    
    return effectiveness


def calculate_performance_metrics(processing_stats):
    """
    Calculate comprehensive performance metrics
    
    Args:
        processing_stats: Processing statistics dictionary
    
    Returns:
        dict: Performance metrics
    """
    metrics = {}
    
    # Processing speed metrics
    if 'total_time' in processing_stats and 'records_saved' in processing_stats:
        total_time = processing_stats['total_time']
        records_saved = processing_stats['records_saved']
        if total_time > 0:
            metrics['overall_processing_speed'] = records_saved / total_time
    
    # Stage-wise performance
    if 'chunk_creation_time' in processing_stats:
        metrics['chunk_creation_speed'] = (
            processing_stats.get('chunks_created', 0) / 
            processing_stats['chunk_creation_time']
        ) if processing_stats['chunk_creation_time'] > 0 else 0
    
    if 'avg_speed' in processing_stats:
        metrics['embedding_speed'] = processing_stats['avg_speed']
    
    # Memory efficiency (if available)
    if 'total_batches' in processing_stats and 'avg_speed' in processing_stats:
        metrics['batch_efficiency'] = processing_stats['avg_speed'] * processing_stats['total_batches']
    
    return metrics


def create_enhanced_run_summary(start_time, end_time, stats, final_analysis, config):
    """
    Create enhanced run summary with comprehensive statistics
    
    Args:
        start_time: Start timestamp
        end_time: End timestamp
        stats: Enhanced processing statistics
        final_analysis: Final analysis results
        config: Configuration object
    
    Returns:
        str: Enhanced formatted summary
    """
    duration = end_time - start_time
    
    summary = []
    summary.append("🚀 ENHANCED RAG INDEXER RUN SUMMARY")
    summary.append("=" * 70)
    summary.append(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"Duration: {duration/60:.1f} minutes ({duration:.1f} seconds)")
    summary.append("")
    
    # Enhanced configuration summary
    summary.extend(create_config_summary(config))
    
    # Main statistics
    summary.extend(create_stats_summary(stats))
    
    # Processing stages
    if stats.get('processing_stages'):
        summary.append("")
        summary.append("🔄 PROCESSING STAGES COMPLETED:")
        for i, stage in enumerate(stats['processing_stages'], 1):
            summary.append(f"   {i}. {stage.replace('_', ' ').title()}")
    
    # Feature-specific summaries
    summary.extend(create_feature_summaries(stats))
    
    # Final analysis summary
    if final_analysis:
        summary.extend(create_final_analysis_summary(final_analysis))
    
    # Failed files summary
    summary.extend(create_failed_files_summary(final_analysis))
    
    summary.append("")
    summary.append("=" * 70)
    
    return "\n".join(summary)


def create_config_summary(config):
    """Create configuration summary section"""
    summary = []
    summary.append("🔧 ENHANCED CONFIGURATION:")
    summary.append(f"   Embedding model: {config.EMBED_MODEL} ({config.EMBED_DIM}D)")
    summary.append(f"   Chunk size: {config.CHUNK_SIZE} (overlap: {config.CHUNK_OVERLAP})")
    summary.append(f"   Processing batch size: {config.PROCESSING_BATCH_SIZE}")
    summary.append(f"   CPU threads: {config.OLLAMA_NUM_THREAD}")
    summary.append(f"   Features enabled:")
    summary.append(f"     - Auto .doc conversion: {'✅' if getattr(config, 'AUTO_CONVERT_DOC', False) else '❌'}")
    summary.append(f"     - OCR processing: {'✅' if config.ENABLE_OCR else '❌'}")
    summary.append(f"     - Safe Ollama restarts: ✅")
    summary.append("")
    return summary


def create_stats_summary(stats):
    """Create main statistics summary section"""
    summary = []
    summary.append("📊 PROCESSING STATISTICS:")
    
    for key, value in stats.items():
        if key in ['rotation_stats', 'quality_analysis_results', 'processing_stages']:
            continue  # Handle these separately
        if isinstance(value, float):
            summary.append(f"   {key}: {value:.2f}")
        elif isinstance(value, int):
            summary.append(f"   {key}: {value:,}")
        else:
            summary.append(f"   {key}: {value}")
    
    summary.append("")
    return summary


def create_feature_summaries(stats):
    """Create feature-specific summary sections"""
    summary = []
    
    # OCR and rotation statistics
    if stats.get('rotation_stats'):
        rotation_stats = stats['rotation_stats']
        summary.append("🔄 AUTO-ROTATION STATISTICS:")
        summary.append(f"   Images tested: {rotation_stats.get('images_tested', 0)}")
        summary.append(f"   Rotations applied: {rotation_stats.get('rotations_applied', 0)}")
        summary.append(f"   Quality improvements: {rotation_stats.get('improvements_found', 0)}")
        if rotation_stats.get('timeouts', 0) > 0:
            summary.append(f"   Timeouts: {rotation_stats['timeouts']}")
        summary.append("")
    
    # Quality analysis results
    if stats.get('quality_analysis_results'):
        quality_results = stats['quality_analysis_results']
        summary.append("🎯 TEXT QUALITY ANALYSIS:")
        summary.append(f"   Filter success rate: {quality_results.get('filter_success_rate', 0):.1f}%")
        summary.append(f"   Invalid chunks filtered: {quality_results.get('invalid_chunks', 0):,}")
        summary.append(f"   Average content length: {quality_results.get('avg_content_length', 0):.0f} chars")
        summary.append("")
    
    return summary


def create_final_analysis_summary(final_analysis):
    """Create final analysis summary section"""
    summary = []
    summary.append("🔍 END-TO-END ANALYSIS:")
    summary.append(f"   Total files in directory: {final_analysis['total_files_in_directory']:,}")
    summary.append(f"   Files successfully in database: {final_analysis['files_successfully_in_db']:,}")
    summary.append(f"   Files missing from database: {final_analysis['files_missing_from_db']:,}")
    summary.append(f"   End-to-end success rate: {final_analysis['success_rate']:.1f}%")
    
    # Feature effectiveness
    if 'feature_effectiveness' in final_analysis:
        effectiveness = final_analysis['feature_effectiveness']
        summary.append("")
        summary.append("✨ FEATURE EFFECTIVENESS:")
        for feature, value in effectiveness.items():
            if isinstance(value, float):
                summary.append(f"   {feature.replace('_', ' ').title()}: {value:.1f}")
            else:
                summary.append(f"   {feature.replace('_', ' ').title()}: {value}")
    
    # Performance metrics
    if 'performance_metrics' in final_analysis:
        metrics = final_analysis['performance_metrics']
        summary.append("")
        summary.append("⚡ PERFORMANCE METRICS:")
        for metric, value in metrics.items():
            summary.append(f"   {metric.replace('_', ' ').title()}: {value:.2f} items/sec")
    
    summary.append("")
    return summary


def create_failed_files_summary(final_analysis):
    """Create failed files summary section"""
    summary = []
    summary.append("❌ FAILED FILES SUMMARY:")
    summary.append("-" * 40)
    
    missing_files_detailed = final_analysis.get('missing_files_detailed', []) if final_analysis else []
    
    if not missing_files_detailed:
        summary.append("✅ No failed files - perfect processing!")
    else:
        summary.append(f"❌ Total failed files: {len(missing_files_detailed)}")
        summary.append(f"📋 Details saved to: /logs/failed_files_details.log")
        
        # Show first 5 for quick reference
        if len(missing_files_detailed) <= 5:
            summary.append("")
            summary.append("All failed files:")
            for i, failed_file in enumerate(missing_files_detailed, 1):
                summary.append(f"  {i}. {failed_file}")
        else:
            summary.append("")
            summary.append("First 5 failed files:")
            for i, failed_file in enumerate(missing_files_detailed[:5], 1):
                summary.append(f"  {i}. {failed_file}")
            summary.append(f"  ... and {len(missing_files_detailed) - 5} more (see detailed log)")
    
    return summary


def create_enhanced_status_report(status_reporter, stats, final_analysis, batch_results, deletion_info, start_time, end_time):
    """
    Create enhanced status report with comprehensive metrics
    
    Args:
        status_reporter: Status reporter instance
        stats: Processing statistics
        final_analysis: Final analysis results
        batch_results: Batch processing results
        deletion_info: Deletion information
        start_time: Processing start time
        end_time: Processing end time
    """
    # Final statistics
    status_reporter.add_section("📊 Enhanced Final Statistics", {
        "Total processing time": f"{end_time - start_time:.2f}s ({(end_time - start_time)/60:.1f}m)",
        "Documents loaded": f"{stats['documents_loaded']:,}",
        "Images processed": f"{stats['images_processed']:,}",
        "Chunks created": f"{stats['chunks_created']:,}",
        "Valid chunks": f"{stats['valid_chunks']:,}",
        "Records saved": f"{stats['records_saved']:,}",
        "Processing speed": f"{batch_results['avg_speed']:.2f} chunks/sec",
        "Overall success rate": f"{batch_results['success_rate']:.1f}%"
    })
    
    # Enhanced features performance
    enhanced_features = {}
    if stats.get('rotation_stats'):
        rotation_stats = stats['rotation_stats']
        enhanced_features["Auto-rotation applied"] = f"{rotation_stats.get('rotations_applied', 0)} images"
        enhanced_features["Quality improvements found"] = f"{rotation_stats.get('improvements_found', 0)}"
    
    if stats.get('advanced_parsing_usage', 0) > 0:
        enhanced_features["Advanced document parsing"] = f"{stats['advanced_parsing_usage']} files"
    
    if stats.get('quality_analysis_results'):
        quality_results = stats['quality_analysis_results']
        enhanced_features["Quality filter success"] = f"{quality_results.get('filter_success_rate', 0):.1f}%"
    
    if enhanced_features:
        status_reporter.add_section("✨ Enhanced Features Performance", enhanced_features)
    
    # End-to-end analysis
    if final_analysis:
        status_reporter.add_section("🔍 End-to-End File Analysis", {
            "Total files in directory": f"{final_analysis['total_files_in_directory']:,}",
            "Files successfully in database": f"{final_analysis['files_successfully_in_db']:,}",
            "Files missing from database": f"{final_analysis['files_missing_from_db']:,}",
            "End-to-end success rate": f"{final_analysis['success_rate']:.1f}%",
            "Missing files details": f"Saved to logs/failed_files_details.log" if final_analysis.get('missing_files_detailed') else "No missing files"
        })
    
    # Data loss analysis
    total_attempted = stats.get('chunks_created', 0)
    total_saved = stats.get('records_saved', 0)
    total_lost = total_attempted - total_saved
    loss_rate = (total_lost / total_attempted * 100) if total_attempted > 0 else 0
    
    status_reporter.add_section("📉 Data Loss Analysis", {
        "Total chunks attempted": f"{total_attempted:,}",
        "Chunks successfully saved": f"{total_saved:,}",
        "Chunks lost in pipeline": f"{total_lost:,}",
        "Pipeline loss rate": f"{loss_rate:.2f}%",
        "Main loss causes": f"See logs/invalid_chunks_report.log for details"
    })
    
    # Quality metrics
    status_reporter.add_section("🎯 Quality Metrics", {
        "Processing pipeline errors": final_analysis.get('files_missing_from_db', 0) if final_analysis else 0,
        "Encoding issues": stats['encoding_issues'],
        "Failed chunks": batch_results['total_failed_chunks'],
        "Failed batches": batch_results['failed_batches'],
        "Embedding errors": batch_results['total_embedding_errors'],
        "Invalid chunks filtered": stats.get('quality_analysis_results', {}).get('invalid_chunks', 0)
    })
    
    # Performance analysis
    performance_data = {
        "Average processing speed": f"{batch_results['avg_speed']:.2f} chunks/sec",
        "Total processing time": f"{batch_results['total_time']:.1f}s"
    }
    
    if final_analysis and 'performance_metrics' in final_analysis:
        metrics = final_analysis['performance_metrics']
        for metric, value in metrics.items():
            performance_data[metric.replace('_', ' ').title()] = f"{value:.2f} items/sec"
    
    status_reporter.add_section("⚡ Performance Analysis", performance_data)
