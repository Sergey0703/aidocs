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
  
  getPipelineStatus: async (taskId = null) => { /* ... */ },
  getPerformanceMetrics: async (taskId = null) => { /* ... */ },
  getErrorLogs: async (options = {}) => { /* ... */ },
  getProcessingQueue: async () => { /* ... */ },
  getChunkAnalysis: async () => { /* ... */ },
  getDatabaseStats: async () => { /* ... */ },
  getMonitoringHealth: async () => { /* ... */ },
  getMetricsSummary: async () => { /* ... */ },
  
  // ============================================================================
  // DOCUMENT & VEHICLE MANAGEMENT ENDPOINTS
  // ============================================================================
  
  getUnassignedAndGroupedDocuments: async () => {
    console.warn("API: getUnassignedAndGroupedDocuments is a mock.");
    return new Promise(resolve => setTimeout(() => resolve({
      grouped: [
        { vrn: '191-D-12345', vehicleExists: true, vehicleDetails: { id: 'uuid-vehicle-1', make: 'Ford', model: 'Focus' }, documents: [{ id: 'doc-uuid-1', filename: 'insurance_cert_2024.pdf' }, { id: 'doc-uuid-2', filename: 'service_invoice_11_2023.pdf' }] },
        { vrn: '241-KY-999', vehicleExists: false, vehicleDetails: null, suggestedMake: 'Toyota', suggestedModel: 'Yaris', documents: [{ id: 'doc-uuid-3', filename: 'purchase_agreement_new_car.docx' }] }
      ],
      unassigned: [{ id: 'doc-uuid-4', filename: 'unrecognized_document.pdf' }, { id: 'doc-uuid-5', filename: 'drivers_manual.pdf' }]
    }), 1000));
  },
  
  linkDocumentsToVehicle: async (vehicleId, documentIds) => {
    console.log(`API: Linking docs [${documentIds.join(', ')}] to vehicle ${vehicleId}`);
    return { success: true, message: `${documentIds.length} documents linked.` };
  },
  
  createVehicleAndLinkDocuments: async (vrn, documentIds, vehicleDetails) => {
    console.log(`API: Creating vehicle ${vrn} with details:`, vehicleDetails, `and linking docs [${documentIds.join(', ')}]`);
    return { success: true, message: `Vehicle ${vrn} created and ${documentIds.length} documents linked.` };
  },
  
  getVehiclesList: async () => {
    console.warn("API: getVehiclesList is a mock.");
    return [
      { id: 'uuid-vehicle-1', registration_number: '191-D-12345', make: 'Ford', model: 'Focus' },
      { id: 'uuid-vehicle-2', registration_number: '211-C-7890', make: 'Toyota', model: 'Corolla' },
      { id: 'uuid-vehicle-3', registration_number: '182-G-4455', make: 'Volkswagen', model: 'Golf' },
    ];
  },

  // ============================================================================
  // VEHICLE CRUD ENDPOINTS (NEW SECTION)
  // ============================================================================

  // Get list of all vehicles for the master list view
  getVehicles: async () => {
    console.warn("API: getVehicles is a mock.");
    // REAL CALL:
    // const response = await api.get('/api/vehicles');
    // return response.data;
    return [
      { id: 'uuid-vehicle-1', registration_number: '191-D-12345', make: 'Ford', model: 'Focus', status: 'active' },
      { id: 'uuid-vehicle-2', registration_number: '211-C-7890', make: 'Toyota', model: 'Corolla', status: 'active' },
      { id: 'uuid-vehicle-3', registration_number: '182-G-4455', make: 'Volkswagen', model: 'Golf', status: 'maintenance' },
    ];
  },

  // Get full details for a single vehicle, including its documents
  getVehicleDetails: async (vehicleId) => {
    console.warn(`API: getVehicleDetails for ${vehicleId} is a mock.`);
    // REAL CALL:
    // const response = await api.get(`/api/vehicles/${vehicleId}`);
    // return response.data;
    return {
      id: vehicleId,
      registration_number: '191-D-12345',
      vin_number: 'WF0XXGCDXXB12345',
      make: 'Ford',
      model: 'Focus',
      insurance_expiry_date: '2025-12-01',
      motor_tax_expiry_date: '2025-08-31',
      nct_expiry_date: '2026-05-20',
      status: 'active',
      documents: [
        { id: 'doc-uuid-1', filename: 'insurance_cert_2024.pdf' },
        { id: 'doc-uuid-2', filename: 'service_invoice_11_2023.pdf' },
      ]
    };
  },

  // Create a new vehicle (without linking documents directly)
  createVehicle: async (vehicleData) => {
    console.log("API: Creating new vehicle", vehicleData);
    // REAL CALL:
    // const response = await api.post('/api/vehicles', vehicleData);
    // return response.data;
    const newVehicle = { ...vehicleData, id: `new-uuid-${Math.random()}`, status: 'active' };
    return newVehicle;
  },

  // Update a vehicle's data
  updateVehicle: async (vehicleId, vehicleData) => {
    console.log(`API: Updating vehicle ${vehicleId}`, vehicleData);
    // REAL CALL:
    // const response = await api.put(`/api/vehicles/${vehicleId}`, vehicleData);
    // return response.data;
    return { ...vehicleData, id: vehicleId };
  },

  // Delete a vehicle
  deleteVehicle: async (vehicleId) => {
    console.log(`API: Deleting vehicle ${vehicleId}`);
    // REAL CALL:
    // const response = await api.delete(`/api/vehicles/${vehicleId}`);
    // return response.data;
    return { success: true, message: 'Vehicle deleted.' };
  },

  // Unlink a document from a vehicle
  unlinkDocumentFromVehicle: async (documentId, vehicleId) => {
    console.log(`API: Unlinking doc ${documentId} from vehicle ${vehicleId}`);
    // REAL CALL:
    // const response = await api.post('/api/manager/unlink', { document_id: documentId, vehicle_id: vehicleId });
    // return response.data;
    return { success: true, message: 'Document unlinked.' };
  }
};

// Export default
export default ragApi;