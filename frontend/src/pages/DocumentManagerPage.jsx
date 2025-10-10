// src/pages/DocumentManagerPage.jsx
import React, { useState, useEffect, useCallback } from 'react';
import './DocumentManagerPage.css';
import { ragApi } from '../api/ragApi';
import GroupedDocuments from '../components/document-manager/GroupedDocuments';
import UnassignedDocuments from '../components/document-manager/UnassignedDocuments';
import FindVRNProgress from '../components/document-manager/FindVRNProgress';
import { FiSearch, FiRefreshCw } from 'react-icons/fi';

const DocumentManagerPage = () => {
  const [groupedDocs, setGroupedDocs] = useState([]);
  const [unassignedDocs, setUnassignedDocs] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // 🆕 STATE FOR VRN FINDING
  const [isFindingVRN, setIsFindingVRN] = useState(false);
  const [findVRNProgress, setFindVRNProgress] = useState(null);
  const [showProgress, setShowProgress] = useState(false);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      console.log('📡 Fetching documents for Document Manager...');
      const data = await ragApi.getUnassignedAndGroupedDocuments();
      console.log('✅ Documents loaded:', data);
      setGroupedDocs(data.grouped || []);
      setUnassignedDocs(data.unassigned || []);
    } catch (err) {
      console.error('❌ Failed to load documents:', err);
      setError("Failed to load documents. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // 🆕 HANDLE FIND VRN
  const handleFindVRN = async () => {
    if (isFindingVRN) {
      console.warn('⚠️ VRN finding already in progress');
      return;
    }

    // Check if there are documents to process
    const totalDocs = unassignedDocs.length;
    if (totalDocs === 0) {
      alert('ℹ️ No unassigned documents to analyze.\n\nPlease upload and index documents first.');
      return;
    }

    // Show confirmation if there are many documents
    if (totalDocs > 20) {
      const confirmed = window.confirm(
        `You are about to analyze ${totalDocs} documents. This may take a few minutes. Continue?`
      );
      if (!confirmed) return;
    }

    setIsFindingVRN(true);
    setShowProgress(true);
    setFindVRNProgress({
      total: totalDocs,
      processed: 0,
      found: 0,
      notFound: 0,
      errors: 0,
      isRunning: true
    });

    try {
      console.log('🔍 Starting VRN extraction for', totalDocs, 'documents');

      // Call backend API
      const result = await ragApi.findVRNInDocuments();

      console.log('✅ VRN extraction completed:', result);

      // Update progress with final results
      setFindVRNProgress({
        total: result.total_processed || totalDocs,
        processed: result.total_processed || 0,
        found: result.vrn_found || 0,
        notFound: result.vrn_not_found || 0,
        errors: result.failed || 0,
        isRunning: false
      });

      // Show detailed success message
      if (result.vrn_found > 0) {
        setTimeout(() => {
          const methods = result.extraction_methods || {};
          const methodsText = `Extraction methods:\n` +
            `  • Regex: ${methods.regex || 0}\n` +
            `  • AI: ${methods.ai || 0}\n` +
            `  • Filename: ${methods.filename || 0}`;

          alert(
            `✅ VRN Extraction Complete!\n\n` +
            `Found VRN in ${result.vrn_found} document(s)\n\n` +
            `Total processed: ${result.total_processed}\n` +
            `VRN found: ${result.vrn_found}\n` +
            `No VRN found: ${result.vrn_not_found}\n` +
            `Errors: ${result.failed || 0}\n\n` +
            `${methodsText}`
          );
        }, 500);
      } else {
        setTimeout(() => {
          alert(
            `ℹ️ No VRN Found\n\n` +
            `Analyzed ${result.total_processed} document(s)\n` +
            `No vehicle registration numbers were detected.\n\n` +
            `You may need to:\n` +
            `  • Check if documents contain VRN\n` +
            `  • Manually assign these documents\n` +
            `  • Verify document quality`
          );
        }, 500);
      }

      // Refresh data after 2 seconds
      setTimeout(() => {
        fetchData();
        setShowProgress(false);
      }, 2000);

    } catch (err) {
      console.error('❌ VRN extraction failed:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Unknown error';
      setError(`Failed to find VRN: ${errorMessage}`);
      
      setFindVRNProgress(prev => ({
        ...prev,
        isRunning: false,
        errors: (prev?.errors || 0) + 1
      }));

      // Show error alert
      setTimeout(() => {
        alert(
          `❌ VRN Extraction Failed\n\n` +
          `Error: ${errorMessage}\n\n` +
          `Please check:\n` +
          `  • Backend is running\n` +
          `  • Documents are indexed\n` +
          `  • Database connection is working`
        );
      }, 500);

      // Hide progress after error
      setTimeout(() => {
        setShowProgress(false);
      }, 3000);
    } finally {
      setIsFindingVRN(false);
    }
  };

  // Handle link to vehicle
  const handleLinkToVehicle = async (vrn, documentIds) => {
    const group = groupedDocs.find(g => g.vrn === vrn);
    if (!group || !group.vehicleDetails) {
      console.error('❌ Vehicle details not found for VRN:', vrn);
      return;
    }

    try {
      console.log('🔗 Batch linking documents to vehicle:', { 
        vrn, 
        vehicleId: group.vehicleDetails.id, 
        documentCount: documentIds.length 
      });

      const result = await ragApi.linkDocumentsToVehicle(group.vehicleDetails.id, documentIds);
      
      console.log('✅ Batch link successful:', result);
      
      setGroupedDocs(prev => prev.filter(g => g.vrn !== vrn));
      
      if (result.failed_ids && result.failed_ids.length > 0) {
        alert(`Linked ${result.linked_count} documents. ${result.failed_ids.length} failed.`);
      }
    } catch (err) {
      console.error('❌ Failed to link documents:', err);
      alert(`Failed to link documents: ${err.message || 'Unknown error'}`);
    }
  };

  // Handle create and link
  const handleCreateAndLink = async (vrn, documentIds, vehicleDetails) => {
    try {
      console.log('🚗 Creating vehicle and linking documents:', { 
        vrn, 
        documentCount: documentIds.length,
        vehicleDetails 
      });

      const result = await ragApi.createVehicleAndLinkDocuments(vrn, documentIds, vehicleDetails);
      
      console.log('✅ Vehicle created and documents linked:', result);
      
      setGroupedDocs(prev => prev.filter(g => g.vrn !== vrn));
      
      if (result.failed_ids && result.failed_ids.length > 0) {
        alert(`Vehicle created! Linked ${result.linked_count} documents. ${result.failed_ids.length} failed.`);
      }
    } catch (err) {
      console.error('❌ Failed to create vehicle and link documents:', err);
      alert(`Failed to create vehicle: ${err.message || 'Unknown error'}`);
    }
  };

  // Handle manual assign
  const handleManualAssign = async (documentId, vehicleId) => {
    if (!documentId || !vehicleId) {
      console.error('❌ Document ID or Vehicle ID missing');
      return;
    }

    try {
      console.log('🔗 Manually assigning document to vehicle:', { documentId, vehicleId });

      await ragApi.linkDocumentToVehicle(vehicleId, documentId);
      
      console.log('✅ Manual assignment successful');
      
      setUnassignedDocs(prev => prev.filter(doc => doc.id !== documentId));
    } catch (err) {
      console.error('❌ Failed to manually assign document:', err);
      alert(`Failed to assign document: ${err.message || 'Unknown error'}`);
    }
  };

  // ============================================================================
  // RENDER
  // ============================================================================

  if (isLoading) {
    return (
      <div className="loading-state">
        <div className="loading-spinner"></div>
        <p>Loading Document Manager...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-state">
        <p className="error-message">{error}</p>
        <button className="retry-button" onClick={fetchData}>
          <FiRefreshCw />
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="doc-manager-container">
      {/* 🆕 HEADER WITH ACTIONS */}
      <div className="doc-manager-header">
        <div className="header-title">
          <h1>Document Manager</h1>
          <p className="header-subtitle">
            Organize and assign documents to vehicles
          </p>
        </div>
        
        <div className="header-actions">
          {/* Refresh Button */}
          <button 
            className="refresh-button"
            onClick={fetchData}
            disabled={isLoading}
            title="Refresh documents"
          >
            <FiRefreshCw className={isLoading ? 'spinning' : ''} />
            <span>Refresh</span>
          </button>

          {/* 🆕 FIND VRN BUTTON - ALWAYS VISIBLE */}
          <button 
            className="find-vrn-button"
            onClick={handleFindVRN}
            disabled={isFindingVRN || unassignedDocs.length === 0}
            title={
              unassignedDocs.length === 0 
                ? 'No unassigned documents to analyze' 
                : `Analyze ${unassignedDocs.length} unassigned document${unassignedDocs.length !== 1 ? 's' : ''}`
            }
          >
            <FiSearch />
            <span>
              {isFindingVRN ? 'Finding VRN...' : 'Find VRN in Documents'}
            </span>
            {unassignedDocs.length > 0 && !isFindingVRN && (
              <span className="doc-count-badge">{unassignedDocs.length}</span>
            )}
          </button>
        </div>
      </div>

      {/* 🆕 PROGRESS DISPLAY */}
      {showProgress && findVRNProgress && (
        <FindVRNProgress progress={findVRNProgress} />
      )}

      {/* STATS BAR */}
      <div className="stats-bar">
        <div className="stat-item">
          <span className="stat-label">Smart Suggestions:</span>
          <span className="stat-value">{groupedDocs.length}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Unassigned:</span>
          <span className="stat-value">{unassignedDocs.length}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Total:</span>
          <span className="stat-value">{groupedDocs.length + unassignedDocs.length}</span>
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className="doc-manager-page">
        {/* LEFT COLUMN - Smart Suggestions */}
        <div className="manager-column">
          <div className="column-header">
            <h2>Smart Suggestions</h2>
            {groupedDocs.length > 0 && (
              <span className="column-count">{groupedDocs.length} group{groupedDocs.length !== 1 ? 's' : ''}</span>
            )}
          </div>
          {groupedDocs.length > 0 ? (
            <div className="grouped-list">
              {groupedDocs.map(group => (
                <GroupedDocuments
                  key={group.vrn}
                  group={group}
                  onLink={handleLinkToVehicle}
                  onCreateAndLink={handleCreateAndLink}
                />
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">✅</div>
              <h3>All documents are processed!</h3>
              <p>No new documents with detected vehicle numbers.</p>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN - Unassigned Documents */}
        <div className="manager-column">
          <div className="column-header">
            <h2>Unassigned Documents</h2>
            {unassignedDocs.length > 0 && (
              <span className="column-count">{unassignedDocs.length} document{unassignedDocs.length !== 1 ? 's' : ''}</span>
            )}
          </div>
          {unassignedDocs.length > 0 ? (
            <UnassignedDocuments 
              documents={unassignedDocs} 
              onAssign={handleManualAssign} 
            />
          ) : (
            <div className="empty-state">
              <div className="empty-icon">🎉</div>
              <h3>Inbox Zero!</h3>
              <p>No documents are waiting for manual assignment.</p>
              <button 
                className="empty-state-action"
                onClick={() => window.location.href = '/indexing'}
              >
                Upload New Documents
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DocumentManagerPage;