// src/components/document-manager/CreateVehicleModal.jsx
import React, { useState, useEffect } from 'react';
import './CreateVehicleModal.css';

const CreateVehicleModal = ({ isOpen, onClose, onSave, vrn, initialData = {} }) => {
  const [formData, setFormData] = useState({
    make: '',
    model: '',
    vin_number: '',
  });

  useEffect(() => {
    // Заполняем форму, когда модальное окно открывается или меняются начальные данные
    if (isOpen) {
      setFormData({
        make: initialData.suggestedMake || '',
        model: initialData.suggestedModel || '',
        vin_number: '',
      });
    }
  }, [isOpen, initialData]);

  if (!isOpen) {
    return null;
  }

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(formData);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Create New Vehicle</h3>
          <button className="modal-close-button" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="vrn">Registration Number</label>
              <input type="text" id="vrn" value={vrn} disabled />
            </div>
            <div className="form-group">
              <label htmlFor="make">Make</label>
              <input
                type="text"
                id="make"
                name="make"
                value={formData.make}
                onChange={handleChange}
                placeholder="e.g., Toyota"
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="model">Model</label>
              <input
                type="text"
                id="model"
                name="model"
                value={formData.model}
                onChange={handleChange}
                placeholder="e.g., Yaris"
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="vin_number">VIN Number (Optional)</label>
              <input
                type="text"
                id="vin_number"
                name="vin_number"
                value={formData.vin_number}
                onChange={handleChange}
                placeholder="Enter VIN"
              />
            </div>
          </form>
        </div>
        <div className="modal-footer">
          <button className="modal-cancel-button" onClick={onClose}>Cancel</button>
          <button className="modal-save-button" onClick={handleSubmit}>Create and Link</button>
        </div>
      </div>
    </div>
  );
};

export default CreateVehicleModal;