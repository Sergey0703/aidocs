// src/pages/IndexingPage.jsx
import React, { useState, useEffect, useCallback } from 'react';
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
  const [indexingResult, setIndexingResult] = useState(null);

  // Documents state
  const [documents, setDocuments] = useState([]);
  const [documentStats, setDocumentStats] = useState(null);
  const [loadingDocuments, setLoadingDocuments] = useState(true);

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
    forceReindex: false,
  });

  // Error state
  const [error, setError] = useState(null);

  const loadDocuments = useCallback(async () => {
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
  }, []);

  const loadDocumentStats = useCallback(async () => {
    try {
      const stats = await ragApi.getDocumentStats();
      setDocumentStats(stats);
    } catch (err) {
      console.error('Failed to load document stats:', err);
    }
  }, []);

  const handleStartIndexing = useCallback(async () => {
    setError(null);
    setIndexingResult(null);
    setIndexingStatus(null);
    setIsIndexing(true);

    try {
      const response = await ragApi.startIndexing({
        mode: indexingSettings.mode,
        skipConversion: true,
        batchSize: parseInt(indexingSettings.batchSize),
        forceReindex: indexingSettings.forceReindex,
      });
      setIndexingTaskId(response.task_id);
    } catch (err) {
      console.error('Failed to start indexing:', err);
      setError(err.response?.data?.detail || 'Failed to start indexing');
      setIsIndexing(false);
    }
  }, [indexingSettings]);

  useEffect(() => {
    let conversionInterval = null;
    let indexingInterval = null;

    if (isConverting && conversionTaskId) {
      conversionInterval = setInterval(async () => {
        try {
          const status = await ragApi.getConversionStatus(conversionTaskId);
          setConversionStatus(status);
          
          const currentStatus = status?.progress?.status;
          if (currentStatus === 'completed' || currentStatus === 'failed') {
            setIsConverting(false);
            if (currentStatus === 'completed' && status.progress.converted_files > 0) {
              setTimeout(handleStartIndexing, 1000);
            }
          }
        } catch (err) {
          console.error('Failed to get conversion status:', err);
          setError('Failed to poll conversion status.');
          setIsConverting(false);
        }
      }, 2000);
    }

    if (isIndexing && indexingTaskId) {
      indexingInterval = setInterval(async () => {
        try {
          const status = await ragApi.getIndexingStatus(indexingTaskId);
          setIndexingStatus(status);
          
          const currentStatus = status?.progress?.status;
          if (currentStatus === 'completed' || currentStatus === 'failed' || currentStatus === 'cancelled') {
            setIsIndexing(false);
            
            if (currentStatus === 'completed') {
              const processed = status.statistics?.documents_processed ?? (status.statistics?.chunks_saved > 0 ? 1 : 0);
              const skipped = status.progress?.skipped_files || 0;

              if (processed === 0 && skipped > 0) {
                setIndexingResult({ type: 'info', message: `Indexing finished. ${skipped} file(s) were already up-to-date and were skipped.` });
              } else {
                setIndexingResult({ type: 'success', message: `Successfully indexed ${processed} file(s) and created ${status.statistics?.chunks_saved || 0} chunks.` });
              }
              
              setTimeout(() => {
                loadDocuments();
                loadDocumentStats();
              }, 500);
            } else {
              setIndexingResult({ type: 'error', message: `Indexing failed. Check logs for details.` });
            }
          }
        } catch (err) {
          console.error('Failed to get indexing status:', err);
          setError('Failed to poll indexing status. Please refresh.');
          setIsIndexing(false);
        }
      }, 2000);
    }

    return () => {
      if (conversionInterval) clearInterval(conversionInterval);
      if (indexingInterval) clearInterval(indexingInterval);
    };
  }, [isConverting, conversionTaskId, isIndexing, indexingTaskId, handleStartIndexing, loadDocuments, loadDocumentStats]);

  useEffect(() => {
    loadDocuments();
    loadDocumentStats();
  }, [loadDocuments, loadDocumentStats]);

  const handleFilesSelected = async (files) => {
    if (files.length === 0) return;

    setError(null);
    setConversionStatus(null);
    setIndexingStatus(null);
    setIndexingResult(null);
    setUploadProgress({ current: 0, total: files.length, uploading: true });

    try {
      console.log(`Starting upload of ${files.length} files...`);
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        setUploadProgress(prev => ({ ...prev, current: i, currentFile: file.name }));
        await ragApi.uploadDocument(file, false);
      }
      
      setUploadProgress(prev => ({ ...prev, current: files.length, uploading: false }));
      console.log('✅ All files uploaded successfully!');
      
      await new Promise(resolve => setTimeout(resolve, 500));
      
      console.log('Starting conversion...');
      setIsConverting(true);
      
      const response = await ragApi.startConversion({
        incremental: conversionSettings.incremental,
        enableOcr: conversionSettings.enableOcr,
        maxFileSizeMb: conversionSettings.maxFileSizeMb
      });

      setConversionTaskId(response.task_id);
      setTimeout(() => setUploadProgress(null), 2000);
      
    } catch (err) {
      console.error('Failed to upload or convert:', err);
      setError(err.message || err.response?.data?.detail || 'Failed to process files');
      setIsConverting(false);
      setUploadProgress(null);
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
    if (!window.confirm(`Are you sure you want to delete "${filename}"?`)) return;
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
        <div className="left-column">
          <section className="upload-section">
            <h2>📤 Document Upload & Conversion</h2>
            <FileUploader
              onFilesSelected={handleFilesSelected}
              disabled={isConverting || isIndexing || (uploadProgress && uploadProgress.uploading)}
              settings={conversionSettings}
              onSettingsChange={setConversionSettings}
            />
          </section>

          {(isConverting || conversionStatus) && (
            <section className="conversion-section">
              <h2>🔄 Conversion Progress</h2>
              <ConversionProgress status={conversionStatus} isActive={isConverting} />
            </section>
          )}

          <section className="indexing-section">
            <h2>🔍 Vector Indexing</h2>
            
            {!isIndexing && indexingResult && (
              <div className={`indexing-result ${indexingResult.type}`}>
                <span className="result-icon">{indexingResult.type === 'success' ? '✅' : (indexingResult.type === 'info' ? '💡' : '❌')}</span>
                <p className="result-message">{indexingResult.message}</p>
                <button className="result-dismiss" onClick={() => setIndexingResult(null)}>×</button>
              </div>
            )}
            
            <div className="indexing-controls">
              <div className="settings-grid">
                <div className="setting-item">
                  <label>Mode:</label>
                  <select value={indexingSettings.mode} onChange={(e) => setIndexingSettings(s => ({ ...s, mode: e.target.value }))} disabled={isIndexing || isConverting}>
                    <option value="incremental">Incremental</option>
                    <option value="full">Full Reindex</option>
                  </select>
                </div>
                <div className="setting-item">
                  <label>Batch Size:</label>
                  <input type="number" min="10" max="200" value={indexingSettings.batchSize} onChange={(e) => setIndexingSettings(s => ({ ...s, batchSize: parseInt(e.target.value) }))} disabled={isIndexing || isConverting} />
                </div>
                <div className="setting-item checkbox">
                  <label>
                    <input type="checkbox" checked={indexingSettings.forceReindex} onChange={(e) => setIndexingSettings(s => ({ ...s, forceReindex: e.target.checked }))} disabled={isIndexing || isConverting} />
                    Force re-index all
                  </label>
                </div>
              </div>

              <button className="start-indexing-button" onClick={handleStartIndexing} disabled={isConverting || isIndexing}>
                {isIndexing ? 'Indexing...' : 'Start Indexing Manually'}
              </button>
            </div>

            {(isIndexing || indexingStatus) && (
              <IndexingProgress status={indexingStatus} isActive={isIndexing} onStop={handleStopIndexing} />
            )}
          </section>

          {(isIndexing || indexingStatus) && (
            <section className="pipeline-section">
              <h2>⚡ Pipeline Stages</h2>
              <PipelineStages status={indexingStatus} />
            </section>
          )}

          {error && (
            <div className="error-display">
              <h3>⚠️ Error</h3>
              <p>{error}</p>
              <button onClick={() => setError(null)}>Dismiss</button>
            </div>
          )}
        </div>

        <div className="right-column">
          <section className="documents-section">
            <div className="documents-header">
              <h2>📚 Indexed Documents</h2>
              <button className="refresh-button" onClick={handleRefreshDocuments} disabled={loadingDocuments}>
                {loadingDocuments ? 'Refreshing...' : '🔄 Refresh'}
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
                  <div className="stat-label">Avg Chunks/Doc</div>
                </div>
              </div>
            )}

            <DocumentsList
              documents={documents}
              loading={loadingDocuments}
              onDelete={handleDeleteDocument}
            />
          </section>
        </div>
      </div>
    </div>
  );
}

export default IndexingPage;