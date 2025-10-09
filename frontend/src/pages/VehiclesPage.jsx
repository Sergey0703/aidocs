// src/pages/VehiclesPage.jsx
import React, { useState, useEffect, useCallback } from 'react';
import './VehiclesPage.css';
import { ragApi } from '../api/ragApi';
import VehicleList from '../components/vehicles/VehicleList';
import VehicleDetail from '../components/vehicles/VehicleDetail';
import CreateVehicleModal from '../components/document-manager/CreateVehicleModal';
import ConfirmationModal from '../components/common/ConfirmationModal';
import { FiSearch } from 'react-icons/fi';

const VehiclesPage = () => {
  const [vehicles, setVehicles] = useState([]);
  const [selectedVehicle, setSelectedVehicle] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // State for modals
  const [isCreateModalOpen, setCreateModalOpen] = useState(false);
  const [isDeleteModalOpen, setDeleteModalOpen] = useState(false);
  const [vehicleToDelete, setVehicleToDelete] = useState(null);

  const fetchVehicles = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await ragApi.getVehicles();
      setVehicles(data);
    } catch (err) {
      setError("Failed to load vehicles.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchVehicles();
  }, [fetchVehicles]);

  const handleSelectVehicle = async (vehicleId) => {
    try {
      const details = await ragApi.getVehicleDetails(vehicleId);
      setSelectedVehicle(details);
    } catch (err) {
      setError("Failed to load vehicle details.");
    }
  };

  const handleCreateVehicle = async (vehicleData) => {
    try {
      const newVehicle = await ragApi.createVehicle(vehicleData);
      setVehicles(prev => [...prev, newVehicle]);
      setCreateModalOpen(false);
    } catch (err) {
      alert("Failed to create vehicle.");
    }
  };

  const handleDeleteRequest = (vehicle) => {
    setVehicleToDelete(vehicle);
    setDeleteModalOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!vehicleToDelete) return;
    try {
      await ragApi.deleteVehicle(vehicleToDelete.id);
      setVehicles(prev => prev.filter(v => v.id !== vehicleToDelete.id));
      if (selectedVehicle?.id === vehicleToDelete.id) {
        setSelectedVehicle(null);
      }
      setDeleteModalOpen(false);
      setVehicleToDelete(null);
    } catch (err) {
      alert("Failed to delete vehicle.");
    }
  };

  const handleUnlinkDocument = async (documentId) => {
    if (!selectedVehicle) return;
    try {
      await ragApi.unlinkDocumentFromVehicle(documentId, selectedVehicle.id);
      // Optimistically update the UI
      setSelectedVehicle(prev => ({
        ...prev,
        documents: prev.documents.filter(doc => doc.id !== documentId)
      }));
    } catch (err) {
      alert("Failed to unlink document.");
    }
  };


  return (
    <>
      <div className="vehicles-page">
        <div className="vehicle-list-column">
          <VehicleList
            vehicles={vehicles}
            onSelectVehicle={handleSelectVehicle}
            selectedVehicleId={selectedVehicle?.id}
            onCreateNew={() => setCreateModalOpen(true)}
            isLoading={isLoading}
          />
        </div>
        <div className="vehicle-detail-column">
          {selectedVehicle ? (
            <VehicleDetail 
              vehicle={selectedVehicle}
              onDelete={handleDeleteRequest}
              onUnlinkDocument={handleUnlinkDocument}
            />
          ) : (
            <div className="placeholder">
              <FiSearch className="placeholder-icon" />
              <h2>Select a vehicle</h2>
              <p>Choose a vehicle from the list to view its details and documents.</p>
            </div>
          )}
        </div>
      </div>

      <CreateVehicleModal
        isOpen={isCreateModalOpen}
        onClose={() => setCreateModalOpen(false)}
        onSave={handleCreateVehicle}
        // No VRN needed when creating from this page
      />

      <ConfirmationModal
        isOpen={isDeleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        onConfirm={handleDeleteConfirm}
        title="Delete Vehicle"
        message={`Are you sure you want to permanently delete ${vehicleToDelete?.registration_number}? This action cannot be undone.`}
      />
    </>
  );
};

export default VehiclesPage;