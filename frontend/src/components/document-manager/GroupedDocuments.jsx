// src/components/document-manager/GroupedDocuments.jsx
import React from 'react';
import './GroupedDocuments.css';
import { FiFileText } from 'react-icons/fi';

const GroupedDocuments = ({ group, onLink, onCreateAndLink }) => {
  const { vrn, vehicleExists, vehicleDetails, documents } = group;

  const handleAction = () => {
    const documentIds = documents.map(doc => doc.id);
    if (vehicleExists) {
      onLink(vrn, documentIds);
    } else {
      onCreateAndLink(vrn, documentIds);
    }
  };

  return (
    <div className="group-card">
      <header className="group-header">
        <div className={`status-indicator ${vehicleExists ? 'exists' : 'new'}`}></div>
        <div>
          <h3 className="group-title">{vrn}</h3>
          <p className="group-subtitle">
            {vehicleExists
              ? `Found existing vehicle: ${vehicleDetails.make} ${vehicleDetails.model}`
              : `This appears to be a new vehicle`}
          </p>
        </div>
      </header>
      <div className="group-body">
        <ul className="doc-list">
          {documents.map(doc => (
            <li key={doc.id} className="doc-item">
              <FiFileText />
              <span>{doc.filename}</span>
            </li>
          ))}
        </ul>
      </div>
      <footer className="group-footer">
        <button
          className={`action-button ${vehicleExists ? 'link' : 'create'}`}
          onClick={handleAction}
        >
          {vehicleExists
            ? `Link ${documents.length} document(s) to existing vehicle`
            : `Create new vehicle and link ${documents.length} document(s)`}
        </button>
      </footer>
    </div>
  );
};

export default GroupedDocuments;