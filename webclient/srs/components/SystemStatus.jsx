// src/components/SystemStatus.jsx
import React, { useState, useEffect } from 'react';
import { ragApi } from '../api/ragApi';
import './SystemStatus.css';

const SystemStatus = () => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchStatus = async () => {
    try {
      setLoading(true);
      const data = await ragApi.getStatus();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading && !status) return <div className="status-loading">Loading status...</div>;
  if (error) return <div className="status-error">Error: {error}</div>;
  if (!status) return null;

  return (
    <div className="system-status">
      <h3>System Status</h3>
      
      {status.hybrid_enabled && (
        <div className="status-badge hybrid-enabled">Hybrid Search Enabled</div>
      )}

      <div className="status-section">
        <h4>Database</h4>
        {status.database.available ? (
          <div className="status-ok">
            <span className="status-icon">✓</span> Connected
            <div className="status-details">
              <div>Documents: {status.database.total_documents}</div>
              <div>Files: {status.database.unique_files}</div>
            </div>
          </div>
        ) : (
          <div className="status-error">
            <span className="status-icon">✗</span> Error
            <div className="error-message">{status.database.error}</div>
          </div>
        )}
      </div>

      <div className="status-section">
        <h4>Embeddings</h4>
        {status.embedding.available ? (
          <div className="status-ok">
            <span className="status-icon">✓</span> Ready
            <div className="status-details">
              <div>Model: {status.embedding.model}</div>
              <div>Dimension: {status.embedding.dimension}</div>
            </div>
          </div>
        ) : (
          <div className="status-error">
            <span className="status-icon">✗</span> Error
            <div className="error-message">{status.embedding.error}</div>
          </div>
        )}
      </div>

      <div className="status-section">
        <h4>Components</h4>
        {Object.entries(status.components).map(([key, value]) => (
          <div key={key} className={value ? "component-ok" : "component-error"}>
            <span className="status-icon">{value ? '✓' : '✗'}</span>
            {key.replace(/_/g, ' ')}
          </div>
        ))}
      </div>

      <button onClick={fetchStatus} className="refresh-button">
        Refresh Status
      </button>
    </div>
  );
};

export default SystemStatus;