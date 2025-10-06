// src/pages/IndexingPage.jsx
import React, { useState, useEffect } from 'react';
import './IndexingPage.css';
import { ragApi } from '../api/ragApi';
import FileUploader from '../components/indexing/FileUploader';
import ConversionProgress from '../components/indexing/ConversionProgress';
import IndexingProgress from '../components/indexing/IndexingProgress';
import PipelineStages from '../components/indexing/PipelineStages';
import DocumentsList from '../components/indexing/DocumentsList';

function IndexingPage() {
  // Conversion state
  const [conversionTaskId, setConversionTaskId] = useState(null);
  const [conversionStatus, setConversionStatus] = useState(null);
  const [isConverting, setIsConverting] = useState(false);

  // Indexing state
  const [indexingTaskId, setIndexingTaskId] = useState(null);
  const [indexingStatus, setIndexingStatus] = useState(null);
  const [isIndexing, setIsIndexing] = useState(false);

  // Documents state
  const [documents, setDocuments] = useState([]);
  const [documentStats, setDocumentStats] = useState(null);
  const [loadingDocuments, setLoadingDocuments] = useState(false);

  // Upload state
  const [uploadProgress, setUploadProgress] = useState(null);

  // Settings
  const [conversionSettings, setConversionSettings] = useState({
    incremental: true,
    enableOcr: true,
    maxFileSizeMb: 50
  });

  const [indexingSettings, setIndexingSettings] = useState({
    mode: 'incremental',
    batchSize: 50,
    deleteExisting: false
  });

  // Error state
  const [error, setError] = useState(null);

  // Polling intervals
  useEffect(() => {
    let conversionInterval = null;
    let indexingInterval = null;

    if (isConverting && conversionTaskId) {
      conversionInterval = setInterval(async () => {
        try {
          const status = await ragApi.getConversionStatus(conversionTaskId);
          setConversionStatus(status);
          
          if (status.progress.status === 'completed' || status.progress.status === 'failed') {
            setIsConverting(false);
            clearInterval(conversionInterval);
            
            // Auto-start indexing after successful conversion
            if (status.progress.status === 'completed' && status.progress.converted_files > 0) {
              setTimeout(() => {
                handleStartIndexing();
              }, 1000);
            }
          }
        } catch (err) {
          console.error('Failed to get conversion status:', err);
        }
      }, 1000);
    }

    if (isIndexing && indexingTaskId) {
      indexingInterval = setInterval(async () => {
        try {
          const status = await ragApi.getIndexingStatus(indexingTaskId);
          setIndexingStatus(status);
          
          if (status.progress.status === 'completed' || status.progress.status === 'failed') {
            setIsIndexing(false);
            clearInterval(indexingInterval);
            
            // Refresh documents list after indexing
            if (status.progress.status === 'completed') {
              setTimeout(() => {
                loadDocuments();
              }, 1000);
            }
          }
        } catch (err) {
          console.error('Failed to get indexing status:', err);
        }
      }, 1000);
    }

    return () => {
      if (conversionInterval) clearInterval(conversionInterval);
      if (indexingInterval) clearInterval(indexingInterval);
    };
  }, [isConverting, conversionTaskId, isIndexing, indexingTaskId]);

  // Load documents on mount
  useEffect(() => {
    loadDocuments();
    loadDocumentStats();
  }, []);

  const loadDocuments = async () => {
    setLoadingDocuments(true);
    try {
      const response = await ragApi.listDocuments({ limit: 50, sortBy: 'indexed_at', order: 'desc' });
      setDocuments(response.documents);
    } catch (err) {
      console.error('Failed to load documents:', err);
      setError('Failed to load documents');
    } finally {
      setLoadingDocuments(false);
    }
  };

  const loadDocumentStats = async () => {
    try {
      const stats = await ragApi.getDocumentStats();
      setDocumentStats(stats);
    } catch (err) {
      console.error('Failed to load document stats:', err);
    }
  };

  const handleFilesSelected = async (files) => {
    if (files.length === 0) return;

    setError(null);
    setUploadProgress({ current: 0, total: files.length, uploading: true });

    try {
      // STEP 1: Upload each file to server
      console.log(`Starting upload of ${files.length} files...`);
      
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        setUploadProgress({ current: i, total: files.length, uploading: true, currentFile: file.name });
        
        try {
          const uploadResult = await ragApi.uploadDocument(file, false); // autoIndex = false
          console.log(`✅ Uploaded ${i + 1}/${files.length}: ${file.name}`, uploadResult);
        } catch (uploadErr) {
          console.error(`❌ Failed to upload ${file.name}:`, uploadErr);
          throw new Error(`Failed to upload ${file.name}: ${uploadErr.response?.data?.detail || uploadErr.message}`);
        }
      }
      
      setUploadProgress({ current: files.length, total: files.length, uploading: false });
      console.log('✅ All files uploaded successfully!');

      // Small delay to ensure files are written to disk
      await new Promise(resolve => setTimeout(resolve, 500));

      // STEP 2: Start conversion of uploaded files
      console.log('Starting conversion...');
      setIsConverting(true);
      
      const response = await ragApi.startConversion({
        incremental: conversionSettings.incremental,
        enableOcr: conversionSettings.enableOcr,
        maxFileSizeMb: conversionSettings.maxFileSizeMb
      });

      setConversionTaskId(response.task_id);
      setConversionStatus(response);
      
      // Clear upload progress after successful conversion start
      setTimeout(() => setUploadProgress(null), 2000);
      
    } catch (err) {
      console.error('Failed to upload or convert:', err);
      setError(err.message || err.response?.data?.detail || 'Failed to process files');
      setIsConverting(false);
      setUploadProgress(null);
    }
  };

  const handleStartIndexing = async () => {
    setError(null);
    setIsIndexing(true);

    try {
      const response = await ragApi.startIndexing({
        mode: indexingSettings.mode,
        skipConversion: true, // We already converted
        skipIndexing: false,  // DO NOT skip indexing!
        batchSize: indexingSettings.batchSize,
        deleteExisting: indexingSettings.deleteExisting
      });

      setIndexingTaskId(response.task_id);
      setIndexingStatus(response);
    } catch (err) {
      console.error('Failed to start indexing:', err);
      setError(err.response?.data?.detail || 'Failed to start indexing');
      setIsIndexing(false);
    }
  };

  const handleStopIndexing = async () => {
    if (!indexingTaskId) return;

    try {
      await ragApi.stopIndexing(indexingTaskId);
      setIsIndexing(false);
    } catch (err) {
      console.error('Failed to stop indexing:', err);
      setError('Failed to stop indexing');
    }
  };

  const handleDeleteDocument = async (filename) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}"?`)) {
      return;
    }

    try {
      await ragApi.deleteDocument(filename);
      loadDocuments();
      loadDocumentStats();
    } catch (err) {
      console.error('Failed to delete document:', err);
      setError('Failed to delete document');
    }
  };

  const handleRefreshDocuments = () => {
    loadDocuments();
    loadDocumentStats();
  };

  return (
    <div className="indexing-page">
      <div className="indexing-page-container">
        {/* Left Column: Upload & Processing */}
        <div className="left-column">
          {/* File Upload Section */}
          <section className="upload-section">
            <h2>📤 Document Upload & Conversion</h2>
            
            {/* Upload Progress Display */}
            {uploadProgress && uploadProgress.uploading && (
              <div style={{
                padding: '1rem',
                background: '#e3f2fd',
                borderRadius: '8px',
                marginBottom: '1rem',
                border: '2px solid #2196f3'
              }}>
                <div style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>
                  Uploading files to server...
                </div>
                <div style={{ fontSize: '0.9rem', color: '#666' }}>
                  {uploadProgress.current} / {uploadProgress.total} files uploaded
                </div>
                {uploadProgress.currentFile && (
                  <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.25rem' }}>
                    Current: {uploadProgress.currentFile}
                  </div>
                )}
                <div style={{
                  marginTop: '0.5rem',
                  height: '8px',
                  background: '#e0e0e0',
                  borderRadius: '4px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${(uploadProgress.current / uploadProgress.total) * 100}%`,
                    height: '100%',
                    background: '#2196f3',
                    transition: 'width 0.3s'
                  }} />
                </div>
              </div>
            )}
            
            {uploadProgress && !uploadProgress.uploading && (
              <div style={{
                padding: '1rem',
                background: '#d4edda',
                borderRadius: '8px',
                marginBottom: '1rem',
                border: '2px solid #28a745',
                color: '#155724'
              }}>
                ✅ All files uploaded successfully! Starting conversion...
              </div>
            )}
            
            <FileUploader
              onFilesSelected={handleFilesSelected}
              disabled={isConverting || isIndexing || (uploadProgress && uploadProgress.uploading)}
              settings={conversionSettings}
              onSettingsChange={setConversionSettings}
            />
          </section>

          {/* Conversion Progress */}
          {(isConverting || conversionStatus) && (
            <section className="conversion-section">
              <h2>🔄 Conversion Progress</h2>
              <ConversionProgress
                status={conversionStatus}
                isActive={isConverting}
              />
            </section>
          )}

          {/* Indexing Controls & Progress */}
          <section className="indexing-section">
            <h2>🔍 Vector Indexing</h2>
            
            {!isIndexing && !indexingStatus && (
              <div className="indexing-controls">
                <div className="settings-grid">
                  <div className="setting-item">
                    <label>Mode:</label>
                    <select
                      value={indexingSettings.mode}
                      onChange={(e) => setIndexingSettings({
                        ...indexingSettings,
                        mode: e.target.value
                      })}
                    >
                      <option value="incremental">Incremental</option>
                      <option value="full">Full Reindex</option>
                    </select>
                  </div>

                  <div className="setting-item">
                    <label>Batch Size:</label>
                    <input
                      type="number"
                      min="10"
                      max="200"
                      value={indexingSettings.batchSize}
                      onChange={(e) => setIndexingSettings({
                        ...indexingSettings,
                        batchSize: parseInt(e.target.value)
                      })}
                    />
                  </div>

                  <div className="setting-item checkbox">
                    <label>
                      <input
                        type="checkbox"
                        checked={indexingSettings.deleteExisting}
                        onChange={(e) => setIndexingSettings({
                          ...indexingSettings,
                          deleteExisting: e.target.checked
                        })}
                      />
                      Delete existing records
                    </label>
                  </div>
                </div>

                <button
                  className="start-indexing-button"
                  onClick={handleStartIndexing}
                  disabled={isConverting || isIndexing}
                >
                  Start Indexing
                </button>
              </div>
            )}

            {(isIndexing || indexingStatus) && (
              <IndexingProgress
                status={indexingStatus}
                isActive={isIndexing}
                onStop={handleStopIndexing}
              />
            )}
          </section>

          {/* Pipeline Visualization */}
          {indexingStatus && (
            <section className="pipeline-section">
              <h2>⚡ Pipeline Stages</h2>
              <PipelineStages status={indexingStatus} />
            </section>
          )}

          {/* Error Display */}
          {error && (
            <div className="error-display">
              <h3>⚠️ Error</h3>
              <p>{error}</p>
              <button onClick={() => setError(null)}>Dismiss</button>
            </div>
          )}
        </div>

        {/* Right Column: Documents List */}
        <div className="right-column">
          <section className="documents-section">
            <div className="documents-header">
              <h2>📚 Indexed Documents</h2>
              <button
                className="refresh-button"
                onClick={handleRefreshDocuments}
                disabled={loadingDocuments}
              >
                🔄 Refresh
              </button>
            </div>

            {documentStats && (
              <div className="stats-summary">
                <div className="stat-card">
                  <div className="stat-value">{documentStats.total_documents}</div>
                  <div className="stat-label">Documents</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{documentStats.total_chunks}</div>
                  <div className="stat-label">Chunks</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{documentStats.avg_chunks_per_document.toFixed(1)}</div>
                  <div className="stat-label">Avg Chunks</div>
                </div>
              </div>
            )}

            <DocumentsList
              documents={documents}
              loading={loadingDocuments}
              onDelete={handleDeleteDocument}
              onRefresh={handleRefreshDocuments}
            />
          </section>
        </div>
      </div>
    </div>
  );
}

export default IndexingPage;