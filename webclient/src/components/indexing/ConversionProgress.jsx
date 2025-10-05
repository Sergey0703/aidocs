// src/components/indexing/ConversionProgress.jsx
import React from 'react';
import './ConversionProgress.css';

const ConversionProgress = ({ status, isActive }) => {
  if (!status) return null;

  const { progress } = status;

  const formatTime = (seconds) => {
    if (!seconds) return 'N/A';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  const getStatusBadge = () => {
    switch (progress.status) {
      case 'pending':
        return <span className="status-badge pending">⏳ Pending</span>;
      case 'converting':
        return <span className="status-badge converting">🔄 Converting</span>;
      case 'completed':
        return <span className="status-badge completed">✅ Completed</span>;
      case 'failed':
        return <span className="status-badge failed">❌ Failed</span>;
      default:
        return <span className="status-badge">{progress.status}</span>;
    }
  };

  const getProgressColor = () => {
    if (progress.failed_files > 0) return '#dc3545';
    if (progress.status === 'completed') return '#28a745';
    return '#007bff';
  };

  return (
    <div className="conversion-progress">
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
          style={{
            width: `${progress.progress_percentage}%`,
            backgroundColor: getProgressColor()
          }}
        >
          {progress.progress_percentage > 5 && (
            <span className="progress-bar-text">
              {progress.converted_files + progress.failed_files + progress.skipped_files} / {progress.total_files}
            </span>
          )}
        </div>
      </div>

      {/* Current File */}
      {progress.current_file && (
        <div className="current-file">
          <span className="current-file-label">Converting:</span>
          <span className="current-file-name">{progress.current_file}</span>
        </div>
      )}

      {/* Statistics Grid */}
      <div className="conversion-stats">
        <div className="stat-item">
          <div className="stat-icon">📊</div>
          <div className="stat-content">
            <div className="stat-label">Total Files</div>
            <div className="stat-value">{progress.total_files}</div>
          </div>
        </div>

        <div className="stat-item success">
          <div className="stat-icon">✅</div>
          <div className="stat-content">
            <div className="stat-label">Converted</div>
            <div className="stat-value">{progress.converted_files}</div>
          </div>
        </div>

        <div className="stat-item error">
          <div className="stat-icon">❌</div>
          <div className="stat-content">
            <div className="stat-label">Failed</div>
            <div className="stat-value">{progress.failed_files}</div>
          </div>
        </div>

        <div className="stat-item skipped">
          <div className="stat-icon">⏩</div>
          <div className="stat-content">
            <div className="stat-label">Skipped</div>
            <div className="stat-value">{progress.skipped_files}</div>
          </div>
        </div>
      </div>

      {/* Time Information */}
      <div className="time-info">
        <div className="time-item">
          <span className="time-label">⏱️ Elapsed:</span>
          <span className="time-value">{formatTime(progress.elapsed_time)}</span>
        </div>
        {progress.estimated_remaining && progress.status === 'converting' && (
          <div className="time-item">
            <span className="time-label">⏳ Remaining:</span>
            <span className="time-value">{formatTime(progress.estimated_remaining)}</span>
          </div>
        )}
      </div>

      {/* Results List (if completed or failed) */}
      {(progress.status === 'completed' || progress.status === 'failed') && status.results && status.results.length > 0 && (
        <div className="conversion-results">
          <h4>Conversion Results:</h4>
          <div className="results-list">
            {status.results.slice(0, 10).map((result, index) => (
              <div key={index} className={`result-item ${result.status}`}>
                <div className="result-icon">
                  {result.status === 'completed' ? '✅' : '❌'}
                </div>
                <div className="result-info">
                  <div className="result-filename">{result.filename}</div>
                  {result.error_message && (
                    <div className="result-error">{result.error_message}</div>
                  )}
                  {result.conversion_time && (
                    <div className="result-time">
                      {formatTime(result.conversion_time)}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {status.results.length > 10 && (
              <div className="results-more">
                ... and {status.results.length - 10} more
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ConversionProgress;