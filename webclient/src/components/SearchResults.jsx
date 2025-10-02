// src/components/SearchResults.jsx
import React, { useState } from 'react';
import DocumentCard from './DocumentCard';
import './SearchResults.css';

const SearchResults = ({ results, answer, totalResults, entityResult, rewriteResult, performanceMetrics }) => {
  const [showEntityDetails, setShowEntityDetails] = useState(false);
  const [showPerformanceDetails, setShowPerformanceDetails] = useState(false);

  if (!answer) {
    return (
      <div className="no-results">
        <p>No results found. Try a different query.</p>
      </div>
    );
  }

  return (
    <div className="search-results">
      {/* ANSWER SECTION - Now at the top */}
      <div className="answer-section">
        <h2>Answer</h2>
        <div className="answer-box">
          {answer.split('\n').map((line, idx) => (
            <p key={idx}>{line}</p>
          ))}
        </div>
      </div>

      {/* COLLAPSIBLE: Smart Entity Extraction */}
      {entityResult && (
        <div className="collapsible-section">
          <button 
            className="collapsible-toggle"
            onClick={() => setShowEntityDetails(!showEntityDetails)}
          >
            {showEntityDetails ? '▼' : '▶'} Smart Entity Extraction
          </button>
          
          {showEntityDetails && (
            <div className="entity-details">
              <div className="detail-row">
                <span className="detail-label">Original Query:</span>
                <span className="detail-value">{entityResult.metadata?.original_query || 'N/A'}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Extracted Entity:</span>
                <span className="detail-value highlight">{entityResult.entity}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Method:</span>
                <span className="badge">{entityResult.method}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Confidence:</span>
                <span className="confidence-score">{(entityResult.confidence * 100).toFixed(1)}%</span>
              </div>
              {rewriteResult && rewriteResult.rewrites && rewriteResult.rewrites.length > 0 && (
                <div className="detail-row">
                  <span className="detail-label">Query Variants ({rewriteResult.rewrites.length}):</span>
                  <div className="query-variants">
                    {rewriteResult.rewrites.map((variant, idx) => (
                      <span key={idx} className="variant-badge">{variant}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* COLLAPSIBLE: Performance Analytics */}
      {performanceMetrics && (
        <div className="collapsible-section">
          <button 
            className="collapsible-toggle"
            onClick={() => setShowPerformanceDetails(!showPerformanceDetails)}
          >
            {showPerformanceDetails ? '▼' : '▶'} Performance Analytics
          </button>
          
          {showPerformanceDetails && (
            <div className="performance-details">
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">Total Time</div>
                  <div className="metric-value">{(performanceMetrics.total_time * 1000).toFixed(0)}ms</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Extraction</div>
                  <div className="metric-value">{(performanceMetrics.extraction_time * 1000).toFixed(0)}ms</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Retrieval</div>
                  <div className="metric-value">{(performanceMetrics.retrieval_time * 1000).toFixed(0)}ms</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Fusion</div>
                  <div className="metric-value">{(performanceMetrics.fusion_time * 1000).toFixed(0)}ms</div>
                </div>
                {performanceMetrics.rerank_time > 0 && (
                  <div className="metric-card">
                    <div className="metric-label">Re-ranking</div>
                    <div className="metric-value">{(performanceMetrics.rerank_time * 1000).toFixed(0)}ms</div>
                  </div>
                )}
                <div className="metric-card">
                  <div className="metric-label">Answer Gen</div>
                  <div className="metric-value">{(performanceMetrics.answer_time * 1000).toFixed(0)}ms</div>
                </div>
              </div>
              
              {performanceMetrics.pipeline_efficiency && (
                <div className="efficiency-breakdown">
                  <h4>Pipeline Efficiency Breakdown</h4>
                  <div className="efficiency-bars">
                    {performanceMetrics.pipeline_efficiency.extraction_pct > 0 && (
                      <div className="efficiency-bar">
                        <span className="bar-label">Extraction</span>
                        <div className="bar-container">
                          <div 
                            className="bar-fill" 
                            style={{width: `${performanceMetrics.pipeline_efficiency.extraction_pct}%`}}
                          ></div>
                          <span className="bar-value">{performanceMetrics.pipeline_efficiency.extraction_pct.toFixed(1)}%</span>
                        </div>
                      </div>
                    )}
                    {performanceMetrics.pipeline_efficiency.rewrite_pct > 0 && (
                      <div className="efficiency-bar">
                        <span className="bar-label">Rewrite</span>
                        <div className="bar-container">
                          <div 
                            className="bar-fill" 
                            style={{width: `${performanceMetrics.pipeline_efficiency.rewrite_pct}%`}}
                          ></div>
                          <span className="bar-value">{performanceMetrics.pipeline_efficiency.rewrite_pct.toFixed(1)}%</span>
                        </div>
                      </div>
                    )}
                    {performanceMetrics.pipeline_efficiency.retrieval_pct > 0 && (
                      <div className="efficiency-bar">
                        <span className="bar-label">Retrieval</span>
                        <div className="bar-container">
                          <div 
                            className="bar-fill" 
                            style={{width: `${performanceMetrics.pipeline_efficiency.retrieval_pct}%`}}
                          ></div>
                          <span className="bar-value">{performanceMetrics.pipeline_efficiency.retrieval_pct.toFixed(1)}%</span>
                        </div>
                      </div>
                    )}
                    {performanceMetrics.pipeline_efficiency.rerank_pct > 0 && (
                      <div className="efficiency-bar">
                        <span className="bar-label">Re-ranking</span>
                        <div className="bar-container">
                          <div 
                            className="bar-fill" 
                            style={{width: `${performanceMetrics.pipeline_efficiency.rerank_pct}%`}}
                          ></div>
                          <span className="bar-value">{performanceMetrics.pipeline_efficiency.rerank_pct.toFixed(1)}%</span>
                        </div>
                      </div>
                    )}
                    {performanceMetrics.pipeline_efficiency.answer_pct > 0 && (
                      <div className="efficiency-bar">
                        <span className="bar-label">Answer</span>
                        <div className="bar-container">
                          <div 
                            className="bar-fill" 
                            style={{width: `${performanceMetrics.pipeline_efficiency.answer_pct}%`}}
                          ></div>
                          <span className="bar-value">{performanceMetrics.pipeline_efficiency.answer_pct.toFixed(1)}%</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* DOCUMENT RESULTS - If any */}
      {results && results.length > 0 && (
        <div className="documents-section">
          <div className="results-header">
            <h2>Source Documents ({totalResults})</h2>
            <div className="quality-badges">
              {results.filter(r => r.similarity_score >= 0.8).length > 0 && (
                <span className="badge high-quality">
                  {results.filter(r => r.similarity_score >= 0.8).length} High Quality
                </span>
              )}
              {results.filter(r => r.similarity_score >= 0.7 && r.similarity_score < 0.8).length > 0 && (
                <span className="badge medium-quality">
                  {results.filter(r => r.similarity_score >= 0.7 && r.similarity_score < 0.8).length} Good
                </span>
              )}
            </div>
          </div>

          <div className="documents-list">
            {results.map((doc, index) => (
              <DocumentCard key={`${doc.document_id}-${index}`} doc={doc} index={index + 1} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SearchResults;