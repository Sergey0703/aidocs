// src/pages/DocumentManagerPage.jsx
import React, { useState, useEffect, useCallback } from 'react';
import './DocumentManagerPage.css';
import { ragApi } from '../api/ragApi';
import GroupedDocuments from '../components/document-manager/GroupedDocuments';
import UnassignedDocuments from '../components/document-manager/UnassignedDocuments';

const DocumentManagerPage = () => {
  const [groupedDocs, setGroupedDocs] = useState([]);
  const [unassignedDocs, setUnassignedDocs] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await ragApi.getUnassignedAndGroupedDocuments();
      setGroupedDocs(data.grouped || []);
      setUnassignedDocs(data.unassigned || []);
    } catch (err) {
      setError("Failed to load documents. Please try again.");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleLinkToVehicle = async (vrn, documentIds) => {
    const group = groupedDocs.find(g => g.vrn === vrn);
    if (!group || !group.vehicleDetails) return;

    try {
      await ragApi.linkDocumentsToVehicle(group.vehicleDetails.id, documentIds);
      // Оптимистичное обновление: удаляем группу из UI
      setGroupedDocs(prev => prev.filter(g => g.vrn !== vrn));
    } catch (err) {
      alert("Failed to link documents.");
    }
  };

  const handleCreateAndLink = async (vrn, documentIds) => {
    try {
      await ragApi.createVehicleAndLinkDocuments(vrn, documentIds);
      setGroupedDocs(prev => prev.filter(g => g.vrn !== vrn));
    } catch (err) {
      alert("Failed to create vehicle and link documents.");
    }
  };

  const handleManualAssign = async (documentId, vehicleId) => {
    if (!documentId || !vehicleId) return;
    try {
      await ragApi.linkDocumentsToVehicle(vehicleId, [documentId]);
      // Удаляем документ из списка нераспределенных
      setUnassignedDocs(prev => prev.filter(doc => doc.id !== documentId));
    } catch (err) {
      alert("Failed to manually assign document.");
    }
  };


  if (isLoading) {
    return <div className="loading-state">Loading Document Manager...</div>;
  }
  if (error) {
    return <div className="error-message">{error}</div>;
  }

  return (
    <div className="doc-manager-page">
      <div className="manager-column">
        <h2>Smart Suggestions</h2>
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
            <h3>All documents are processed!</h3>
            <p>No new documents with detected vehicle numbers.</p>
          </div>
        )}
      </div>

      <div className="manager-column">
        <h2>Unassigned Documents</h2>
        {unassignedDocs.length > 0 ? (
          <UnassignedDocuments 
            documents={unassignedDocs} 
            onAssign={handleManualAssign} 
          />
        ) : (
          <div className="empty-state">
            <h3>Inbox Zero!</h3>
            <p>No documents are waiting for manual assignment.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentManagerPage;