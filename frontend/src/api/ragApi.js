// src/api/ragApi.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for better error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

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
    if (!taskId) {
      throw new Error('Task ID is required');
    }
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
    if (!taskId) {
      throw new Error('Task ID is required');
    }
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
    if (!taskId) {
      throw new Error('Task ID is required');
    }
    const response = await api.post(`/api/conversion/retry?task_id=${taskId}`);
    return response.data;
  },

  // Delete conversion task
  deleteConversionTask: async (taskId) => {
    if (!taskId) {
      throw new Error('Task ID is required');
    }
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
    if (!taskId) {
      throw new Error('Task ID is required');
    }
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
    if (!filenames || filenames.length === 0) {
      throw new Error('At least one filename is required');
    }
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
    if (!filename) {
      throw new Error('Filename is required');
    }
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
    if (!filename) {
      throw new Error('Filename is required');
    }
    const response = await api.delete(`/api/documents/${encodeURIComponent(filename)}`, {
      params: { delete_chunks: deleteChunks }
    });
    return response.data;
  },

  // Get document chunks
  getDocumentChunks: async (filename, limit = 100, offset = 0) => {
    if (!filename) {
      throw new Error('Filename is required');
    }
    const response = await api.get(`/api/documents/${encodeURIComponent(filename)}/chunks`, {
      params: { limit, offset }
    });
    return response.data;
  },

  // Upload document
  uploadDocument: async (file, autoIndex = true) => {
    if (!file) {
      throw new Error('File is required');
    }
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
  
  // ============================================================================
  // DOCUMENT & VEHICLE MANAGEMENT ENDPOINTS (NEW SECTION)
  // ============================================================================
  
  // Get all documents that are not yet linked to a vehicle.
  // The backend should return them grouped by any detected registration number.
  getUnassignedAndGroupedDocuments: async () => {
    // MOCK: This endpoint needs to be implemented on the backend.
    console.warn("API: getUnassignedAndGroupedDocuments is a mock.");
    // We return mock data here to allow frontend development.
    return new Promise(resolve => setTimeout(() => resolve({
      grouped: [
        {
          vrn: '191-D-12345',
          vehicleExists: true,
          vehicleDetails: { id: 'uuid-vehicle-1', make: 'Ford', model: 'Focus' },
          documents: [
            { id: 'doc-uuid-1', filename: 'insurance_cert_2024.pdf' },
            { id: 'doc-uuid-2', filename: 'service_invoice_11_2023.pdf' },
          ]
        },
        {
          vrn: '241-KY-999',
          vehicleExists: false,
          vehicleDetails: null,
          // This data is expected to be extracted by the backend from the document content.
          suggestedMake: 'Toyota', 
          suggestedModel: 'Yaris',
          documents: [
            { id: 'doc-uuid-3', filename: 'purchase_agreement_new_car.docx' },
          ]
        }
      ],
      unassigned: [
        { id: 'doc-uuid-4', filename: 'unrecognized_document.pdf' },
        { id: 'doc-uuid-5', filename: 'drivers_manual.pdf' },
      ]
    }), 1000)); // Simulate network delay
    
    /*
    // REAL CALL (once backend is ready):
    const response = await api.get('/api/manager/unassigned');
    return response.data;
    */
  },

  // Link one or more documents to an existing vehicle
  linkDocumentsToVehicle: async (vehicleId, documentIds) => {
    console.log(`API: Linking docs [${documentIds.join(', ')}] to vehicle ${vehicleId}`);
    /*
    // REAL CALL:
    const response = await api.post('/api/manager/link', { vehicle_id: vehicleId, document_ids: documentIds });
    return response.data;
    */
    return { success: true, message: `${documentIds.length} documents linked.` };
  },
  
  // Create a new vehicle and link documents to it
  createVehicleAndLinkDocuments: async (vrn, documentIds, vehicleDetails) => {
    console.log(`API: Creating vehicle ${vrn} with details:`, vehicleDetails);
    console.log(`API: and linking docs [${documentIds.join(', ')}]`);
    /*
    // REAL CALL:
    const response = await api.post('/api/manager/create-and-link', { 
      vrn: vrn, 
      document_ids: documentIds,
      details: vehicleDetails // 'make', 'model', 'vin_number'
    });
    return response.data;
    */
    return { success: true, message: `Vehicle ${vrn} created and ${documentIds.length} documents linked.` };
  },

  // Get a list of all vehicles for manual assignment dropdowns
  getVehiclesList: async () => {
    console.warn("API: getVehiclesList is a mock.");
    /*
    // REAL CALL:
    const response = await api.get('/api/vehicles/list');
    return response.data;
    */
    return [
      { id: 'uuid-vehicle-1', registration_number: '191-D-12345', make: 'Ford', model: 'Focus' },
      { id: 'uuid-vehicle-2', registration_number: '211-C-7890', make: 'Toyota', model: 'Corolla' },
      { id: 'uuid-vehicle-3', registration_number: '182-G-4455', make: 'Volkswagen', model: 'Golf' },
    ];
  },
};

// Export default
export default ragApi;