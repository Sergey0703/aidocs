// src/components/document-manager/GroupedDocuments.jsx
import React, { useState } from 'react';
import './GroupedDocuments.css';
import { FiFileText } from 'react-icons/fi';
import CreateVehicleModal from './CreateVehicleModal'; // <-- ИМПОРТ

const GroupedDocuments = ({ group, onLink, onCreateAndLink }) => {
  const { vrn, vehicleExists, vehicleDetails, documents } = group;
  
  // <-- НОВОЕ СОСТОЯНИЕ ДЛЯ УПРАВЛЕНИЯ МОДАЛЬНЫМ ОКНОМ -->
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleAction = () => {
    const documentIds = documents.map(doc => doc.id);
    if (vehicleExists) {
      // Если машина существует, сразу привязываем
      onLink(vrn, documentIds);
    } else {
      // Если машина новая, открываем модальное окно
      setIsModalOpen(true);
    }
  };

  const handleSaveVehicle = (vehicleData) => {
    // Эта функция вызывается из модального окна при сохранении
    const documentIds = documents.map(doc => doc.id);
    onCreateAndLink(vrn, documentIds, vehicleData);
    setIsModalOpen(false); // Закрываем модальное окно после сохранения
  };

  return (
    <>
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

      {/* <-- РЕНДЕРИНГ МОДАЛЬНОГО ОКНА --> */}
      {!vehicleExists && (
        <CreateVehicleModal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          onSave={handleSaveVehicle}
          vrn={vrn}
          initialData={group} // Передаем всю группу, чтобы получить suggestedMake/Model
        />
      )}
    </>
  );
};

export default GroupedDocuments;