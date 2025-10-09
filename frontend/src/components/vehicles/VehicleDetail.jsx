// src/components/vehicles/VehicleDetail.jsx
import React, { useState } from 'react';
import './VehicleDetail.css';
import { FiFileText, FiEdit, FiTrash2 } from 'react-icons/fi';

const VehicleDetail = ({ vehicle, onDelete, onUnlinkDocument }) => {
  const [activeTab, setActiveTab] = useState('details');

  if (!vehicle) return null;

  return (
    <div className="detail-container">
      <header className="detail-header">
        <div className="detail-title">
          <h2>{vehicle.registration_number}</h2>
          <p>{vehicle.make} {vehicle.model}</p>
        </div>
        <div className="header-actions">
          <button className="edit-btn"><FiEdit /> Edit</button>
          <button className="delete-btn" onClick={() => onDelete(vehicle)}><FiTrash2 /> Delete</button>
        </div>
      </header>

      <div className="detail-tabs">
        <button className={`tab-btn ${activeTab === 'details' ? 'active' : ''}`} onClick={() => setActiveTab('details')}>
          Details
        </button>
        <button className={`tab-btn ${activeTab === 'documents' ? 'active' : ''}`} onClick={() => setActiveTab('documents')}>
          Documents ({vehicle.documents?.length || 0})
        </button>
      </div>

      <div className="detail-body">
        {activeTab === 'details' && (
          <div className="info-grid">
            <div className="info-item"><label>VIN Number</label><span>{vehicle.vin_number || 'N/A'}</span></div>
            <div className="info-item"><label>Status</label><span>{vehicle.status}</span></div>
            <div className="info-item"><label>Insurance Expiry</label><span>{vehicle.insurance_expiry_date || 'N/A'}</span></div>
            <div className="info-item"><label>Motor Tax Expiry</label><span>{vehicle.motor_tax_expiry_date || 'N/A'}</span></div>
            <div className="info-item"><label>NCT Expiry</label><span>{vehicle.nct_expiry_date || 'N/A'}</span></div>
          </div>
        )}

        {activeTab === 'documents' && (
          <div className="doc-list-detail">
            {vehicle.documents?.length > 0 ? vehicle.documents.map(doc => (
              <div key={doc.id} className="doc-item-detail">
                <span className="doc-name"><FiFileText /> {doc.filename}</span>
                <button className="unlink-btn" onClick={() => onUnlinkDocument(doc.id)}>Unlink</button>
              </div>
            )) : <p>No documents linked to this vehicle.</p>}
          </div>
        )}
      </div>
    </div>
  );
};

export default VehicleDetail;