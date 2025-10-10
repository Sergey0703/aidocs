// src/components/document-manager/CreateVehicleModal.jsx
import React, { useState, useEffect } from 'react';
import './CreateVehicleModal.css';

const CreateVehicleModal = ({ isOpen, onClose, onSave, vrn, initialData = {}, isLoading = false }) => {
  const [formData, setFormData] = useState({
    registration_number: '',
    make: '',
    model: '',
    vin_number: '',
    insurance_expiry_date: '',
    motor_tax_expiry_date: '',
    nct_expiry_date: '',
    status: 'active',
  });

  const [errors, setErrors] = useState({});

  // Reset form when modal opens/closes or initialData changes
  useEffect(() => {
    if (isOpen) {
      setFormData({
        registration_number: vrn || '',
        make: initialData.suggestedMake || initialData.make || '',
        model: initialData.suggestedModel || initialData.model || '',
        vin_number: initialData.vin_number || '',
        insurance_expiry_date: initialData.insurance_expiry_date || '',
        motor_tax_expiry_date: initialData.motor_tax_expiry_date || '',
        nct_expiry_date: initialData.nct_expiry_date || '',
        status: initialData.status || 'active',
      });
      setErrors({});
    }
  }, [isOpen]);

  if (!isOpen) {
    return null;
  }

  // ========================================================================
  // FORM HANDLING
  // ========================================================================

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    // Clear error for this field when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: null }));
    }
  };

  const validateForm = () => {
    const newErrors = {};

    // Registration number is required
    if (!formData.registration_number || !formData.registration_number.trim()) {
      newErrors.registration_number = 'Registration number is required';
    }

    // Make is required
    if (!formData.make || !formData.make.trim()) {
      newErrors.make = 'Make is required';
    }

    // Model is required
    if (!formData.model || !formData.model.trim()) {
      newErrors.model = 'Model is required';
    }

    // VIN validation (if provided, must be 17 characters)
    if (formData.vin_number && formData.vin_number.trim().length > 0) {
      if (formData.vin_number.trim().length !== 17) {
        newErrors.vin_number = 'VIN must be exactly 17 characters';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    // Prepare data for submission (remove empty strings)
    const submitData = {
      registration_number: formData.registration_number.trim(),
      make: formData.make.trim(),
      model: formData.model.trim(),
      vin_number: formData.vin_number.trim() || null,
      insurance_expiry_date: formData.insurance_expiry_date || null,
      motor_tax_expiry_date: formData.motor_tax_expiry_date || null,
      nct_expiry_date: formData.nct_expiry_date || null,
      status: formData.status,
    };

    onSave(submitData);
  };

  // ========================================================================
  // RENDER
  // ========================================================================

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <h3>{vrn ? `Create Vehicle: ${vrn}` : 'Create New Vehicle'}</h3>
          <button 
            className="modal-close-button" 
            onClick={onClose}
            disabled={isLoading}
          >
            &times;
          </button>
        </div>

        {/* Body */}
        <div className="modal-body">
          <form onSubmit={handleSubmit} id="create-vehicle-form">
            {/* Registration Number */}
            <div className="form-group">
              <label htmlFor="registration_number">
                Registration Number <span className="required">*</span>
              </label>
              <input
                type="text"
                id="registration_number"
                name="registration_number"
                value={formData.registration_number}
                onChange={handleChange}
                placeholder="e.g., 191-D-12345"
                disabled={!!vrn || isLoading}
                className={errors.registration_number ? 'input-error' : ''}
              />
              {errors.registration_number && (
                <span className="error-message">{errors.registration_number}</span>
              )}
            </div>

            {/* Make */}
            <div className="form-group">
              <label htmlFor="make">
                Make <span className="required">*</span>
              </label>
              <input
                type="text"
                id="make"
                name="make"
                value={formData.make}
                onChange={handleChange}
                placeholder="e.g., Toyota"
                disabled={isLoading}
                className={errors.make ? 'input-error' : ''}
              />
              {errors.make && (
                <span className="error-message">{errors.make}</span>
              )}
            </div>

            {/* Model */}
            <div className="form-group">
              <label htmlFor="model">
                Model <span className="required">*</span>
              </label>
              <input
                type="text"
                id="model"
                name="model"
                value={formData.model}
                onChange={handleChange}
                placeholder="e.g., Yaris"
                disabled={isLoading}
                className={errors.model ? 'input-error' : ''}
              />
              {errors.model && (
                <span className="error-message">{errors.model}</span>
              )}
            </div>

            {/* VIN Number */}
            <div className="form-group">
              <label htmlFor="vin_number">VIN Number (Optional)</label>
              <input
                type="text"
                id="vin_number"
                name="vin_number"
                value={formData.vin_number}
                onChange={handleChange}
                placeholder="17-character VIN"
                maxLength={17}
                disabled={isLoading}
                className={errors.vin_number ? 'input-error' : ''}
              />
              {errors.vin_number && (
                <span className="error-message">{errors.vin_number}</span>
              )}
              <span className="field-hint">Must be exactly 17 characters</span>
            </div>

            {/* Status */}
            <div className="form-group">
              <label htmlFor="status">Status</label>
              <select
                id="status"
                name="status"
                value={formData.status}
                onChange={handleChange}
                disabled={isLoading}
              >
                <option value="active">Active</option>
                <option value="maintenance">Maintenance</option>
                <option value="inactive">Inactive</option>
                <option value="sold">Sold</option>
                <option value="archived">Archived</option>
              </select>
            </div>

            {/* Expiry Dates Section */}
            <div className="form-section-header">
              <h4>Expiry Dates (Optional)</h4>
            </div>

            {/* Insurance Expiry */}
            <div className="form-group">
              <label htmlFor="insurance_expiry_date">Insurance Expiry Date</label>
              <input
                type="date"
                id="insurance_expiry_date"
                name="insurance_expiry_date"
                value={formData.insurance_expiry_date}
                onChange={handleChange}
                disabled={isLoading}
              />
            </div>

            {/* Motor Tax Expiry */}
            <div className="form-group">
              <label htmlFor="motor_tax_expiry_date">Motor Tax Expiry Date</label>
              <input
                type="date"
                id="motor_tax_expiry_date"
                name="motor_tax_expiry_date"
                value={formData.motor_tax_expiry_date}
                onChange={handleChange}
                disabled={isLoading}
              />
            </div>

            {/* NCT Expiry */}
            <div className="form-group">
              <label htmlFor="nct_expiry_date">NCT Expiry Date</label>
              <input
                type="date"
                id="nct_expiry_date"
                name="nct_expiry_date"
                value={formData.nct_expiry_date}
                onChange={handleChange}
                disabled={isLoading}
              />
            </div>
          </form>
        </div>

        {/* Footer */}
        <div className="modal-footer">
          <button 
            className="modal-cancel-button" 
            onClick={onClose}
            disabled={isLoading}
          >
            Cancel
          </button>
          <button 
            className="modal-save-button" 
            onClick={handleSubmit}
            disabled={isLoading}
            form="create-vehicle-form"
          >
            {isLoading ? 'Creating...' : (vrn ? 'Create and Link' : 'Create Vehicle')}
          </button>
        </div>
      </div>
    </div>
  );
};

export default CreateVehicleModal;