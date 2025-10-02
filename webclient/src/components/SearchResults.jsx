// src/components/SearchResults.jsx
import React, { useState } from 'react';
import DocumentCard from './DocumentCard';
import './SearchResults.css';

const SearchResults = ({ results, answer, totalResults, entityResult, rewriteResult, performanceMetrics }) => {
  const [showEntityDetails, setShowEntityDetails] = useState(false);
  const [showPerformanceDetails, setShowPerformanceDetails] = useState(false);

  const formatTime = (seconds) => {
    if (!seconds) return 'N/A';
    return seconds < 1 ? `${(seconds * 1000).toFixed(0)}ms` : `${seconds.toFixed(2)}s`;
  };

  if (!answer) {
    return (
      <div className="no-results">
        <p>No results found. Try a different query.</p>
      </div>
    );
  }

  return (
    <div className="search-results">
      {/* ANSWER SECTION - Always visible at the top */}
      <div className="answer-section">
        <h2>Answer</h2>
        <div className="answer-box">
          {answer.split('\n').map((line, idx) => (
            <p key={idx}>{line}</p>
          ))}
        </div>
      </div>

      {/* SOURCE DOCUMENTS - Always visible after answer */}
      {results && results.length > 0 && (
        <div className="documents-section">
          <div className="results-header">
            <h2>Source Documents ({totalResults})</h2>
            <div className="quality-badges">
              {results.filter(r => r.similarity_score >= 0.8).length > 0 && (
                <span className="quality-badge excellent">
                  {results.filter(r => r.similarity_score >= 0.8).length} High Quality
                </span>
              )}
              {results.filter(r => r.similarity_score >= 0.6 && r.similarity_score < 0.8).length > 0 && (
                <span className="quality-badge good">
                  {results.filter(r => r.similarity_score >= 0.6 && r.similarity_score < 0.8).length} Good
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

      {/* COLLAPSIBLE: Smart Entity Extraction - Collapsed by default */}
      {entityResult && (
        <div className="collapsible-section">
          <button 
            className="collapsible-toggle"
            onClick={() => setShowEntityDetails(!showEntityDetails)}
          >
            <span className="toggle-icon">{showEntityDetails ? '▼' : '▶'}</span>
            <span className="toggle-title">Smart Entity Extraction</span>
            <span className="toggle-badge entity-badge">
              {entityResult.entity} • {(entityResult.confidence * 100).toFixed(0)}%
            </span>
          </button>
          
          {showEntityDetails && (
            <div className="collapsible-content">
              <div className="entity-details">
                <div className="detail-row">
                  <span className="detail-label">Original Query:</span>
                  <span className="detail-value">{rewriteResult?.original_query || 'N/A'}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Extracted Entity:</span>
                  <span className="detail-value highlight">{entityResult.entity}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Method:</span>
                  <span className="detail-badge">{entityResult.method}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Confidence:</span>
                  <span className="confidence-score">{(entityResult.confidence * 100).toFixed(1)}%</span>
                </div>
                {entityResult.alternatives && entityResult.alternatives.length > 0 && (
                  <div className="detail-row">
                    <span className="detail-label">Alternatives:</span>
                    <span className="detail-value alternatives">{entityResult.alternatives.join(', ')}</span>
                  </div>
                )}
                {rewriteResult && rewriteResult.rewrites && rewriteResult.rewrites.length > 0 && (
                  <div className="detail-row variants-row">
                    <span className="detail-label">Query Variants ({rewriteResult.rewrites.length}):</span>
                    <div className="query-variants">
                      {rewriteResult.rewrites.map((variant, idx) => (
                        <span key={idx} className="variant-badge">{variant}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* COLLAPSIBLE: Performance Analytics - Collapsed by default with compact view */}
      {performanceMetrics && (
        <div className="collapsible-section performance-section">
          <button 
            className="collapsible-toggle performance-toggle"
            onClick={() => setShowPerformanceDetails(!showPerformanceDetails)}
          >
            <span className="toggle-icon">{showPerformanceDetails ? '▼' : '▶'}</span>
            <span className="toggle-title">Performance Analytics</span>
            <span className="toggle-badge performance-badge">
              ⚡ {formatTime(performanceMetrics.total_time)} | {
                [
                  performanceMetrics.extraction_time > 0 ? 'Extract' : null,
                  performanceMetrics.rewrite_time > 0 ? 'Rewrite' : null,
                  performanceMetrics.retrieval_time > 0 ? 'Retrieve' : null,
                  performanceMetrics.fusion_time > 0 ? 'Fuse' : null,
                  performanceMetrics.rerank_time > 0 ? 'Rerank' : null,
                  performanceMetrics.answer_time > 0 ? 'Answer' : null
                ].filter(Boolean).length
              } stages
            </span>
          </button>
          
          {showPerformanceDetails && (
            <div className="collapsible-content">
              <div className="performance-details">
                <div className="metrics-grid">
                  <div className="metric-card">
                    <div className="metric-label">Total Time</div>
                    <div className="metric-value">{formatTime(performanceMetrics.total_time)}</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Extraction</div>
                    <div className="metric-value">{formatTime(performanceMetrics.extraction_time)}</div>
                  </div>
                  {performanceMetrics.rewrite_time > 0 && (
                    <div className="metric-card">
                      <div className="metric-label">Rewrite</div>
                      <div className="metric-value">{formatTime(performanceMetrics.rewrite_time)}</div>
                    </div>
                  )}
                  <div className="metric-card">
                    <div className="metric-label">Retrieval</div>
                    <div className="metric-value">{formatTime(performanceMetrics.retrieval_time)}</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Fusion</div>
                    <div className="metric-value">{formatTime(performanceMetrics.fusion_time)}</div>
                  </div>
                  {performanceMetrics.rerank_time > 0 && (
                    <div className="metric-card">
                      <div className="metric-label">Re-Ranking</div>
                      <div className="metric-value">{formatTime(performanceMetrics.rerank_time)}</div>
                    </div>
                  )}
                  <div className="metric-card">
                    <div className="metric-label">Answer Gen</div>
                    <div className="metric-value">{formatTime(performanceMetrics.answer_time)}</div>
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
                              className="bar-fill extraction" 
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
                              className="bar-fill rewrite" 
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
                              className="bar-fill retrieval" 
                              style={{width: `${performanceMetrics.pipeline_efficiency.retrieval_pct}%`}}
                            ></div>
                            <span className="bar-value">{performanceMetrics.pipeline_efficiency.retrieval_pct.toFixed(1)}%</span>
                          </div>
                        </div>
                      )}
                      {performanceMetrics.pipeline_efficiency.rerank_pct > 0 && (
                        <div className="efficiency-bar">
                          <span className="bar-label">Re-Ranking</span>
                          <div className="bar-container">
                            <div 
                              className="bar-fill rerank" 
                              style={{width: `${performanceMetrics.pipeline_efficiency.rerank_pct}%`}}
                            ></div>
                            <span className="bar-value">{performanceMetrics.pipeline_efficiency.rerank_pct.toFixed(1)}%</span>
                          </div>
                        </div>
                      )}
                      {performanceMetrics.pipeline_efficiency.fusion_pct > 0 && (
                        <div className="efficiency-bar">
                          <span className="bar-label">Fusion</span>
                          <div className="bar-container">
                            <div 
                              className="bar-fill fusion" 
                              style={{width: `${performanceMetrics.pipeline_efficiency.fusion_pct}%`}}
                            ></div>
                            <span className="bar-value">{performanceMetrics.pipeline_efficiency.fusion_pct.toFixed(1)}%</span>
                          </div>
                        </div>
                      )}
                      {performanceMetrics.pipeline_efficiency.answer_pct > 0 && (
                        <div className="efficiency-bar">
                          <span className="bar-label">Answer</span>
                          <div className="bar-container">
                            <div 
                              className="bar-fill answer" 
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
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SearchResults;