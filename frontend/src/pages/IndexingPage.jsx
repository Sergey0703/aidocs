// src/pages/IndexingPage.jsx
import React, { useState, useEffect, useCallback } from 'react';
import { ragApi } from '../api/ragApi';
import FileUploader from '../components/indexing/FileUploader';
import ConversionProgress from '../components/indexing/ConversionProgress';
import IndexingProgress from '../components/indexing/IndexingProgress';
import DocumentsList from '../components/indexing/DocumentsList';
import './IndexingPage.css';

const IndexingPage = () => {
  const [conversionStatus, setConversionStatus] = useState(null);
  const [indexingStatus, setIndexingStatus] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [loadingDocs, setLoadingDocs] = useState(true);
  
  const [conversionTaskId, setConversionTaskId] = useState(null);
  const [indexingTaskId, setIndexingTaskId] = useState(null);
  
  const [uploadSettings, setUploadSettings] = useState({
    incremental: true,
    enableOcr: true,
    maxFileSizeMb: 50,
  });

  const fetchDocuments = useCallback(async () => {
    setLoadingDocs(true);
    try {
      const data = await ragApi.listDocuments({ limit: 1000 });
      setDocuments(data.documents || []);
    } catch (error) {
      console.error("Failed to fetch documents:", error);
    } finally {
      setLoadingDocs(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);
  
  // Polling for statuses
  useEffect(() => {
    const pollStatus = async (taskId, getStatusFunc, setStatusFunc, setTaskIdFunc) => {
      if (!taskId) return;
      try {
        const data = await getStatusFunc(taskId);
        setStatusFunc(data);
        const statusValue = data?.progress?.status;
        if (['completed', 'failed', 'cancelled'].includes(statusValue)) {
          setTaskIdFunc(null); // Stop polling
          fetchDocuments(); // Refresh documents list on completion
        }
      } catch (error) {
        console.error("Polling error:", error);
        setTaskIdFunc(null); // Stop polling on error
      }
    };

    const intervalId = setInterval(() => {
      pollStatus(conversionTaskId, ragApi.getConversionStatus, setConversionStatus, setConversionTaskId);
      pollStatus(indexingTaskId, ragApi.getIndexingStatus, setIndexingStatus, setIndexingTaskId);
    }, 2000);

    return () => clearInterval(intervalId);
  }, [conversionTaskId, indexingTaskId, fetchDocuments]);
  
  const handleFilesSelected = async (files) => {
    // This is where you would handle the actual upload logic
    console.log("Uploading files:", files);
    console.log("With settings:", uploadSettings);
    // For now, let's simulate starting a conversion
    try {
      const response = await ragApi.startConversion(uploadSettings);
      setConversionTaskId(response.task_id);
    } catch (error) {
      console.error("Failed to start conversion:", error);
    }
  };

  const handleStartIndexing = async () => {
    try {
      const response = await ragApi.startIndexing({ mode: 'incremental' });
      setIndexingTaskId(response.task_id);
    } catch (error) {
      console.error("Failed to start indexing:", error);
    }
  };
  
  const handleDeleteDocument = async (filename) => {
    try {
      await ragApi.deleteDocument(filename);
      // Refresh list after deletion
      setDocuments(prev => prev.filter(doc => doc.filename !== filename));
    } catch (error) {
      console.error("Failed to delete document:", error);
    }
  };

  return (
    <div className="indexing-page">
      <div className="indexing-left-column">
        {/* Document Upload & Conversion Card */}
        <div className="card">
          <div className="card-header">
            <h3>Document Upload & Conversion</h3>
          </div>
          <div className="card-body">
            <FileUploader
              onFilesSelected={handleFilesSelected}
              disabled={!!conversionTaskId}
              settings={uploadSettings}
              onSettingsChange={setUploadSettings}
            />
          </div>
        </div>

        {/* Conversion Progress Card */}
        {conversionTaskId && conversionStatus && (
          <div className="card">
            <div className="card-header">
              <h3>Conversion Progress</h3>
            </div>
            <div className="card-body">
              <ConversionProgress status={conversionStatus} isActive={!!conversionTaskId} />
            </div>
          </div>
        )}
      </div>

      <div className="indexing-right-column">
        {/* Indexing Control Card */}
        <div className="card">
          <div className="card-header">
            <h3>Vector Indexing</h3>
          </div>
          <div className="card-body">
            <p>Process converted markdown files into searchable vectors.</p>
            <button
              className="start-indexing-button"
              onClick={handleStartIndexing}
              disabled={!!indexingTaskId}
            >
              {indexingTaskId ? 'Indexing in Progress...' : 'Start Incremental Indexing'}
            </button>
            {indexingStatus && (
              <IndexingProgress status={indexingStatus} isActive={!!indexingTaskId} />
            )}
          </div>
        </div>

        {/* Indexed Documents Card */}
        <div className="card">
          <div className="card-header">
            <h3>Indexed Documents</h3>
            <button className="refresh-button" onClick={fetchDocuments} disabled={loadingDocs}>
              {loadingDocs ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
          <div className="card-body">
            <DocumentsList
              documents={documents}
              loading={loadingDocs}
              onDelete={handleDeleteDocument}
              onRefresh={fetchDocuments}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default IndexingPage;