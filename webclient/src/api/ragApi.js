// src/api/ragApi.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const ragApi = {
  // ============================================================================
  // SEARCH ENDPOINTS
  // ============================================================================
  
  // Search endpoint
  search: async (query, maxResults = 20, rerankMode = 'smart') => {
    const response = await api.post('/api/search', {
      query,
      max_results: maxResults,
      rerank_mode: rerankMode,
    });
    return response.data;
  },

  // System status
  getStatus: async () => {
    const response = await api.get('/api/system/status');
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await api.get('/api/system/health');
    return response.data;
  },

  // ============================================================================
  // CONVERSION ENDPOINTS
  // ============================================================================
  
  // Start document conversion
  startConversion: async (options = {}) => {
    const response = await api.post('/api/conversion/start', {
      input_dir: options.inputDir || null,
      output_dir: options.outputDir || null,
      incremental: options.incremental !== false,
      formats: options.formats || null,
      enable_ocr: options.enableOcr || null,
      max_file_size_mb: options.maxFileSizeMb || null,
    });
    return response.data;
  },

  // Get conversion status
  getConversionStatus: async (taskId) => {
    const response = await api.get('/api/conversion/status', {
      params: { task_id: taskId }
    });
    return response.data;
  },

  // Get supported formats
  getSupportedFormats: async () => {
    const response = await api.get('/api/conversion/formats');
    return response.data;
  },

  // Get conversion results
  getConversionResults: async (taskId, includeFailed = true, includeSkipped = false) => {
    const response = await api.get('/api/conversion/results', {
      params: {
        task_id: taskId,
        include_failed: includeFailed,
        include_skipped: includeSkipped
      }
    });
    return response.data;
  },

  // Validate documents
  validateDocuments: async (options = {}) => {
    const response = await api.post('/api/conversion/validate', {
      input_dir: options.inputDir || null,
      check_formats: options.checkFormats !== false,
      check_size: options.checkSize !== false,
      max_size_mb: options.maxSizeMb || null,
    });
    return response.data;
  },

  // Retry failed conversions
  retryFailedConversions: async (taskId) => {
    const response = await api.post(`/api/conversion/retry?task_id=${taskId}`);
    return response.data;
  },

  // Delete conversion task
  deleteConversionTask: async (taskId) => {
    const response = await api.delete(`/api/conversion/task/${taskId}`);
    return response.data;
  },

  // Get conversion history
  getConversionHistory: async (limit = 10) => {
    const response = await api.get('/api/conversion/history', {
      params: { limit }
    });
    return response.data;
  },

  // ============================================================================
  // INDEXING ENDPOINTS
  // ============================================================================
  
  // Start indexing
  startIndexing: async (options = {}) => {
    const response = await api.post('/api/indexing/start', {
      mode: options.mode || 'incremental',
      documents_dir: options.documentsDir || null,
      skip_conversion: options.skipConversion || false,
      skip_indexing: options.skipIndexing || false,
      batch_size: options.batchSize || null,
      force_reindex: options.forceReindex || false,
      delete_existing: options.deleteExisting || false,
    });
    return response.data;
  },

  // Stop indexing
  stopIndexing: async (taskId) => {
    const response = await api.post('/api/indexing/stop', null, {
      params: { task_id: taskId }
    });
    return response.data;
  },

  // Get indexing status
  getIndexingStatus: async (taskId = null) => {
    const params = taskId ? { task_id: taskId } : {};
    const response = await api.get('/api/indexing/status', { params });
    return response.data;
  },

  // Get indexing history
  getIndexingHistory: async (limit = 10) => {
    const response = await api.get('/api/indexing/history', {
      params: { limit }
    });
    return response.data;
  },

  // Clear index
  clearIndex: async (confirm = false) => {
    const response = await api.delete('/api/indexing/clear', {
      params: { confirm }
    });
    return response.data;
  },

  // Reindex specific files
  reindexFiles: async (filenames, force = false) => {
    const response = await api.post('/api/indexing/reindex', {
      filenames,
      force
    });
    return response.data;
  },

  // Get all indexing tasks
  getAllIndexingTasks: async () => {
    const response = await api.get('/api/indexing/tasks');
    return response.data;
  },

  // Cleanup completed tasks
  cleanupCompletedTasks: async () => {
    const response = await api.delete('/api/indexing/tasks/cleanup');
    return response.data;
  },

  // ============================================================================
  // DOCUMENTS ENDPOINTS
  // ============================================================================
  
  // List documents
  listDocuments: async (options = {}) => {
    const response = await api.get('/api/documents', {
      params: {
        limit: options.limit || 100,
        offset: options.offset || 0,
        sort_by: options.sortBy || 'indexed_at',
        order: options.order || 'desc'
      }
    });
    return response.data;
  },

  // Get document details
  getDocument: async (filename, includeChunks = false) => {
    const response = await api.get(`/api/documents/${encodeURIComponent(filename)}`, {
      params: { include_chunks: includeChunks }
    });
    return response.data;
  },

  // Get document statistics
  getDocumentStats: async () => {
    const response = await api.get('/api/documents/stats/overview');
    return response.data;
  },

  // Search documents
  searchDocuments: async (criteria) => {
    const response = await api.post('/api/documents/search', criteria);
    return response.data;
  },

  // Delete document
  deleteDocument: async (filename, deleteChunks = true) => {
    const response = await api.delete(`/api/documents/${encodeURIComponent(filename)}`, {
      params: { delete_chunks: deleteChunks }
    });
    return response.data;
  },

  // Get document chunks
  getDocumentChunks: async (filename, limit = 100, offset = 0) => {
    const response = await api.get(`/api/documents/${encodeURIComponent(filename)}/chunks`, {
      params: { limit, offset }
    });
    return response.data;
  },

  // Upload document
  uploadDocument: async (file, autoIndex = true) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('auto_index', autoIndex);

    const response = await api.post('/api/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Get missing documents
  getMissingDocuments: async () => {
    const response = await api.get('/api/documents/missing/files');
    return response.data;
  },

  // ============================================================================
  // MONITORING ENDPOINTS
  // ============================================================================
  
  // Get pipeline status
  getPipelineStatus: async (taskId = null) => {
    const params = taskId ? { task_id: taskId } : {};
    const response = await api.get('/api/monitoring/pipeline', { params });
    return response.data;
  },

  // Get performance metrics
  getPerformanceMetrics: async (taskId = null) => {
    const params = taskId ? { task_id: taskId } : {};
    const response = await api.get('/api/monitoring/performance', { params });
    return response.data;
  },

  // Get error logs
  getErrorLogs: async (options = {}) => {
    const response = await api.get('/api/monitoring/errors', {
      params: {
        limit: options.limit || 50,
        error_type: options.errorType || null,
        since: options.since || null
      }
    });
    return response.data;
  },

  // Get processing queue
  getProcessingQueue: async () => {
    const response = await api.get('/api/monitoring/queue');
    return response.data;
  },

  // Get chunk analysis
  getChunkAnalysis: async () => {
    const response = await api.get('/api/monitoring/chunks/analysis');
    return response.data;
  },

  // Get database stats
  getDatabaseStats: async () => {
    const response = await api.get('/api/monitoring/database/stats');
    return response.data;
  },

  // Health check (monitoring)
  getMonitoringHealth: async () => {
    const response = await api.get('/api/monitoring/health');
    return response.data;
  },

  // Get metrics summary
  getMetricsSummary: async () => {
    const response = await api.get('/api/monitoring/metrics/summary');
    return response.data;
  },
};