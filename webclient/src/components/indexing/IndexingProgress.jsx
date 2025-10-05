// src/components/indexing/IndexingProgress.jsx
import React from 'react';
import './IndexingProgress.css';

const IndexingProgress = ({ status, isActive, onStop }) => {
  if (!status) return null;

  const { progress, statistics } = status;

  const formatTime = (seconds) => {
    if (!seconds) return 'N/A';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  const getStatusBadge = () => {
    switch (progress.status) {
      case 'idle':
        return <span className="status-badge idle">⏸️ Idle</span>;
      case 'running':
        return <span className="status-badge running">🔄 Running</span>;
      case 'completed':
        return <span className="status-badge completed">✅ Completed</span>;
      case 'failed':
        return <span className="status-badge failed">❌ Failed</span>;
      case 'cancelled':
        return <span className="status-badge cancelled">⏹️ Cancelled</span>;
      default:
        return <span className="status-badge">{progress.status}</span>;
    }
  };

  const getStageIcon = (stage) => {
    switch (stage) {
      case 'conversion': return '🔄';
      case 'loading': return '📂';
      case 'chunking': return '🧩';
      case 'embedding': return '🤖';
      case 'saving': return '💾';
      case 'completed': return '✅';
      default: return '📝';
    }
  };

  const getCurrentStageDisplay = () => {
    if (!progress.stage) return 'Initializing...';
    const stageName = progress.current_stage_name || progress.stage;
    return `${getStageIcon(progress.stage)} ${stageName}`;
  };

  return (
    <div className="indexing-progress">
      {/* Status Header */}
      <div className="progress-header">
        <div className="status-info">
          {getStatusBadge()}
          {isActive && (
            <span className="pulse-indicator">
              <span className="pulse-dot"></span>
              Active
            </span>
          )}
        </div>
        <div className="progress-percentage">
          {progress.progress_percentage.toFixed(1)}%
        </div>
      </div>

      {/* Progress Bar */}
      <div className="progress-bar-container">
        <div
          className="progress-bar-fill"
          style={{ width: `${progress.progress_percentage}%` }}
        >
          {progress.progress_percentage > 5 && (
            <span className="progress-bar-text">
              {getCurrentStageDisplay()}
            </span>
          )}
        </div>
      </div>

      {/* Current Processing Info */}
      {progress.current_file && (
        <div className="current-file">
          <span className="current-file-label">Processing:</span>
          <span className="current-file-name">{progress.current_file}</span>
        </div>
      )}

      {/* Files & Chunks Progress */}
      <div className="processing-stats">
        <div className="stat-group">
          <div className="stat-header">📄 Files</div>
          <div className="stat-progress">
            <span className="stat-current">{progress.processed_files}</span>
            <span className="stat-separator">/</span>
            <span className="stat-total">{progress.total_files}</span>
          </div>
          {progress.failed_files > 0 && (
            <div className="stat-failed">❌ {progress.failed_files} failed</div>
          )}
        </div>

        <div className="stat-group">
          <div className="stat-header">🧩 Chunks</div>
          <div className="stat-progress">
            <span className="stat-current">{progress.processed_chunks}</span>
            <span className="stat-separator">/</span>
            <span className="stat-total">{progress.total_chunks}</span>
          </div>
          {progress.processing_speed > 0 && (
            <div className="stat-speed">
              ⚡ {progress.processing_speed.toFixed(1)} chunks/s
            </div>
          )}
        </div>

        {progress.current_batch && progress.total_batches && (
          <div className="stat-group">
            <div className="stat-header">📦 Batches</div>
            <div className="stat-progress">
              <span className="stat-current">{progress.current_batch}</span>
              <span className="stat-separator">/</span>
              <span className="stat-total">{progress.total_batches}</span>
            </div>
          </div>
        )}
      </div>

      {/* Time Information */}
      <div className="time-info">
        <div className="time-item">
          <span className="time-label">⏱️ Elapsed:</span>
          <span className="time-value">{formatTime(progress.elapsed_time)}</span>
        </div>
        {progress.estimated_remaining && progress.status === 'running' && (
          <div className="time-item">
            <span className="time-label">⏳ Remaining:</span>
            <span className="time-value">{formatTime(progress.estimated_remaining)}</span>
          </div>
        )}
        {progress.avg_time_per_file > 0 && (
          <div className="time-item">
            <span className="time-label">📊 Avg/File:</span>
            <span className="time-value">{formatTime(progress.avg_time_per_file)}</span>
          </div>
        )}
      </div>

      {/* Statistics (if available) */}
      {statistics && (
        <div className="indexing-statistics">
          <h4>📊 Statistics</h4>
          <div className="stats-grid">
            {statistics.documents_loaded > 0 && (
              <div className="stat-card">
                <div className="stat-card-label">Documents Loaded</div>
                <div className="stat-card-value">{statistics.documents_loaded}</div>
              </div>
            )}
            {statistics.chunks_created > 0 && (
              <div className="stat-card">
                <div className="stat-card-label">Chunks Created</div>
                <div className="stat-card-value">{statistics.chunks_created}</div>
              </div>
            )}
            {statistics.chunks_saved > 0 && (
              <div className="stat-card">
                <div className="stat-card-label">Chunks Saved</div>
                <div className="stat-card-value">{statistics.chunks_saved}</div>
              </div>
            )}
            {statistics.success_rate > 0 && (
              <div className="stat-card success">
                <div className="stat-card-label">Success Rate</div>
                <div className="stat-card-value">{(statistics.success_rate * 100).toFixed(1)}%</div>
              </div>
            )}
            {statistics.gemini_api_calls > 0 && (
              <div className="stat-card api">
                <div className="stat-card-label">Gemini API Calls</div>
                <div className="stat-card-value">{statistics.gemini_api_calls}</div>
              </div>
            )}
            {statistics.gemini_tokens_used > 0 && (
              <div className="stat-card api">
                <div className="stat-card-label">Tokens Used</div>
                <div className="stat-card-value">{statistics.gemini_tokens_used.toLocaleString()}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Stop Button (if running) */}
      {isActive && onStop && (
        <button className="stop-button" onClick={onStop}>
          ⏹️ Stop Indexing
        </button>
      )}

      {/* Errors & Warnings */}
      {status.errors && status.errors.length > 0 && (
        <div className="errors-section">
          <h4>❌ Errors ({status.errors.length})</h4>
          <div className="errors-list">
            {status.errors.slice(0, 5).map((error, index) => (
              <div key={index} className="error-item">
                {error}
              </div>
            ))}
            {status.errors.length > 5 && (
              <div className="errors-more">
                ... and {status.errors.length - 5} more errors
              </div>
            )}
          </div>
        </div>
      )}

      {status.warnings && status.warnings.length > 0 && (
        <div className="warnings-section">
          <h4>⚠️ Warnings ({status.warnings.length})</h4>
          <div className="warnings-list">
            {status.warnings.slice(0, 3).map((warning, index) => (
              <div key={index} className="warning-item">
                {warning}
              </div>
            ))}
            {status.warnings.length > 3 && (
              <div className="warnings-more">
                ... and {status.warnings.length - 3} more warnings
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default IndexingProgress;