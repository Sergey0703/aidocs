// src/components/SystemStatus.jsx
import React, { useState, useEffect } from 'react';
import { ragApi } from '../api/ragApi';
import './SystemStatus.css';

const SystemStatus = ({ lastSearchMetrics }) => {
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

  const formatTime = (seconds) => {
    if (!seconds) return 'N/A';
    return seconds < 1 ? `${(seconds * 1000).toFixed(0)}ms` : `${seconds.toFixed(2)}s`;
  };

  if (loading && !status) return <div className="status-loading">Loading status...</div>;
  if (error) return <div className="status-error">Error: {error}</div>;
  if (!status) return null;

  return (
    <div className="system-status">
      <h3>System Status</h3>
      
      {status.hybrid_enabled && (
        <div className="status-badge hybrid-enabled">✅ Hybrid Search Enabled</div>
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

      {lastSearchMetrics && (
        <div className="status-section last-search-section">
          <h4>📊 Last Search</h4>
          <div className="last-search-stats">
            <div className="stat-item">
              <span className="stat-label">Total:</span>
              <span className="stat-value total">{formatTime(lastSearchMetrics.total_time)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Extraction:</span>
              <span className="stat-value">{formatTime(lastSearchMetrics.extraction_time)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Retrieval:</span>
              <span className="stat-value">{formatTime(lastSearchMetrics.retrieval_time)}</span>
            </div>
            {lastSearchMetrics.rerank_time > 0 && (
              <div className="stat-item">
                <span className="stat-label">Re-Ranking:</span>
                <span className="stat-value">{formatTime(lastSearchMetrics.rerank_time)}</span>
              </div>
            )}
            <div className="stat-item">
              <span className="stat-label">Fusion:</span>
              <span className="stat-value">{formatTime(lastSearchMetrics.fusion_time)}</span>
            </div>
          </div>
        </div>
      )}

      <button onClick={fetchStatus} className="refresh-button">
        Refresh Status
      </button>
    </div>
  );
};

export default SystemStatus;