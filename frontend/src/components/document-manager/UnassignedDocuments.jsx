// src/components/document-manager/UnassignedDocuments.jsx
import React, { useState } from 'react';
import './UnassignedDocuments.css';
import { FiFile } from 'react-icons/fi';
import VehicleSearchInput from './VehicleSearchInput'; // 🆕 НОВЫЙ КОМПОНЕНТ

const UnassignedDocuments = ({ documents, onAssign }) => {
  const [assigningFor, setAssigningFor] = useState(null); // ID of doc being assigned

  const handleStartAssign = (docId) => {
    setAssigningFor(docId);
  };

  const handleCancelAssign = () => {
    setAssigningFor(null);
  };

  // 🆕 ОБРАБОТЧИК ВЫБОРА МАШИНЫ ИЗ АВТОКОМПЛИТА
  const handleVehicleSelect = (vehicle) => {
    if (assigningFor && vehicle) {
      console.log('📎 Assigning document to vehicle:', { docId: assigningFor, vehicleId: vehicle.id });
      
      // Вызываем callback родительского компонента
      onAssign(assigningFor, vehicle.id);
      
      // Сбрасываем форму после назначения
      setAssigningFor(null);
    }
  };

  return (
    <div className="unassigned-list">
      {documents.map(doc => (
        <div key={doc.id} className="unassigned-item">
          <div className="unassigned-info">
            <span className="unassigned-name">
              <FiFile /> {doc.filename}
            </span>
            {assigningFor !== doc.id && (
              <button 
                className="assign-button" 
                onClick={() => handleStartAssign(doc.id)}
              >
                Manual Assign
              </button>
            )}
          </div>

          {/* 🆕 ФОРМА С АВТОКОМПЛИТОМ ВМЕСТО SELECT */}
          {assigningFor === doc.id && (
            <div className="assign-form">
              <VehicleSearchInput
                onSelect={handleVehicleSelect}
                placeholder="Type to search vehicles..."
                autoFocus={true}
              />
              <button 
                className="cancel-assign-button" 
                onClick={handleCancelAssign}
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default UnassignedDocuments;