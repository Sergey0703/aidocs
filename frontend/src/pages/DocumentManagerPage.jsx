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

  // 🔄 ОБНОВЛЕННЫЙ: Теперь использует batch endpoint через ragApi
  const handleLinkToVehicle = async (vrn, documentIds) => {
    const group = groupedDocs.find(g => g.vrn === vrn);
    if (!group || !group.vehicleDetails) {
      console.error('❌ Vehicle details not found for VRN:', vrn);
      return;
    }

    try {
      console.log('📎 Batch linking documents to vehicle:', { 
        vrn, 
        vehicleId: group.vehicleDetails.id, 
        documentCount: documentIds.length 
      });

      // 🆕 Теперь использует новый batch метод (внутри ragApi)
      const result = await ragApi.linkDocumentsToVehicle(group.vehicleDetails.id, documentIds);
      
      console.log('✅ Batch link successful:', result);
      
      // Удаляем группу из списка после успешной привязки
      setGroupedDocs(prev => prev.filter(g => g.vrn !== vrn));
      
      // Показываем уведомление пользователю
      if (result.failed_ids && result.failed_ids.length > 0) {
        alert(`Linked ${result.linked_count} documents. ${result.failed_ids.length} failed.`);
      }
    } catch (err) {
      console.error('❌ Failed to link documents:', err);
      alert(`Failed to link documents: ${err.message || 'Unknown error'}`);
    }
  };

  // 🔄 ОБНОВЛЕННЫЙ: Теперь использует create-and-link endpoint через ragApi
  const handleCreateAndLink = async (vrn, documentIds, vehicleDetails) => {
    try {
      console.log('🚗 Creating vehicle and linking documents:', { 
        vrn, 
        documentCount: documentIds.length,
        vehicleDetails 
      });

      // 🆕 Теперь использует новый create-and-link метод (внутри ragApi)
      const result = await ragApi.createVehicleAndLinkDocuments(vrn, documentIds, vehicleDetails);
      
      console.log('✅ Vehicle created and documents linked:', result);
      
      // Удаляем группу из списка после успешного создания
      setGroupedDocs(prev => prev.filter(g => g.vrn !== vrn));
      
      // Показываем уведомление пользователю
      if (result.failed_ids && result.failed_ids.length > 0) {
        alert(`Vehicle created! Linked ${result.linked_count} documents. ${result.failed_ids.length} failed.`);
      }
    } catch (err) {
      console.error('❌ Failed to create vehicle and link documents:', err);
      alert(`Failed to create vehicle: ${err.message || 'Unknown error'}`);
    }
  };

  // ✅ БЕЗ ИЗМЕНЕНИЙ: Manual assign уже работает правильно
  const handleManualAssign = async (documentId, vehicleId) => {
    if (!documentId || !vehicleId) {
      console.error('❌ Document ID or Vehicle ID missing');
      return;
    }

    try {
      console.log('📎 Manually assigning document to vehicle:', { documentId, vehicleId });

      // Используем single link метод для ручного назначения
      await ragApi.linkDocumentToVehicle(vehicleId, documentId);
      
      console.log('✅ Manual assignment successful');
      
      // Удаляем документ из списка неназначенных
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
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="doc-manager-page">
      {/* LEFT COLUMN - Smart Suggestions (Grouped by VRN) */}
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

      {/* RIGHT COLUMN - Unassigned Documents */}
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