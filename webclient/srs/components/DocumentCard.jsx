// src/components/DocumentCard.jsx
import React, { useState } from 'react';
import './DocumentCard.css';

const DocumentCard = ({ doc, index }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const getSourceIcon = (method) => {
    if (method.includes('database')) return '🗄️';
    if (method.includes('vector')) return '🔍';
    return '📄';
  };

  const getQualityClass = (score) => {
    if (score >= 0.8) return 'excellent';
    if (score >= 0.6) return 'good';
    if (score >= 0.4) return 'moderate';
    return 'low';
  };

  const getSourceLabel = (method) => {
    if (method.includes('database')) return 'Database Match';
    if (method.includes('vector')) return 'Vector Match';
    return 'Search Result';
  };

  return (
    <div className={`document-card ${getQualityClass(doc.similarity_score)}`}>
      <div className="card-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="card-title">
          <span className="doc-number">{index}.</span>
          <span className="source-icon">{getSourceIcon(doc.source_method)}</span>
          <span className="filename">{doc.filename}</span>
          <span className="similarity-score">
            {(doc.similarity_score * 100).toFixed(1)}%
          </span>
        </div>
        <button className="expand-button">
          {isExpanded ? '▼' : '▶'}
        </button>
      </div>

      {isExpanded && (
        <div className="card-content">
          <div className="content-preview">
            <h4>Content Preview:</h4>
            <p>{doc.content}</p>
          </div>

          <div className="document-metadata">
            <h4>Document Intelligence:</h4>
            <div className="metadata-grid">
              <div className="metadata-item">
                <span className="metadata-label">Similarity:</span>
                <span className="metadata-value">{doc.similarity_score.toFixed(3)}</span>
              </div>
              <div className="metadata-item">
                <span className="metadata-label">Source:</span>
                <span className="metadata-value">{getSourceLabel(doc.source_method)}</span>
              </div>
              <div className="metadata-item">
                <span className="metadata-label">Method:</span>
                <span className="metadata-value">{doc.source_method}</span>
              </div>
              {doc.chunk_index > 0 && (
                <div className="metadata-item">
                  <span className="metadata-label">Chunk:</span>
                  <span className="metadata-value">{doc.chunk_index}</span>
                </div>
              )}
            </div>

            {doc.metadata && Object.keys(doc.metadata).length > 0 && (
              <div className="additional-metadata">
                <h5>Additional Information:</h5>
                {Object.entries(doc.metadata).slice(0, 5).map(([key, value]) => (
                  <div key={key} className="metadata-item">
                    <span className="metadata-label">{key.replace(/_/g, ' ')}:</span>
                    <span className="metadata-value">{String(value).substring(0, 100)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentCard;