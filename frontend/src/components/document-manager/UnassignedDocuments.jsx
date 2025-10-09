// src/components/document-manager/UnassignedDocuments.jsx
import React, { useState, useEffect } from 'react';
import './UnassignedDocuments.css';
import { FiFile } from 'react-icons/fi';
import { ragApi } from '../../api/ragApi';

const UnassignedDocuments = ({ documents, onAssign }) => {
  const [assigningFor, setAssigningFor] = useState(null); // ID of doc being assigned
  const [selectedVehicle, setSelectedVehicle] = useState('');
  const [vehicles, setVehicles] = useState([]);

  useEffect(() => {
    // Загружаем список машин один раз при монтировании
    const fetchVehicles = async () => {
      const vehicleList = await ragApi.getVehiclesList();
      setVehicles(vehicleList);
    };
    fetchVehicles();
  }, []);

  const handleStartAssign = (docId) => {
    setAssigningFor(docId);
    setSelectedVehicle('');
  };

  const handleSaveAssign = () => {
    if (assigningFor && selectedVehicle) {
      onAssign(assigningFor, selectedVehicle);
      setAssigningFor(null); // Сбрасываем форму после сохранения
    }
  };

  return (
    <div className="unassigned-list">
      {documents.map(doc => (
        <div key={doc.id} className="unassigned-item">
          <div className="unassigned-info">
            <span className="unassigned-name"><FiFile /> {doc.filename}</span>
            {assigningFor !== doc.id && (
              <button className="assign-button" onClick={() => handleStartAssign(doc.id)}>
                Manual Assign
              </button>
            )}
          </div>

          {assigningFor === doc.id && (
            <div className="assign-form">
              <select
                value={selectedVehicle}
                onChange={(e) => setSelectedVehicle(e.target.value)}
              >
                <option value="" disabled>-- Select a vehicle --</option>
                {vehicles.map(v => (
                  <option key={v.id} value={v.id}>
                    {v.registration_number} ({v.make} {v.model})
                  </option>
                ))}
              </select>
              <button className="save-assign-button" onClick={handleSaveAssign}>Save</button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default UnassignedDocuments;