// src/components/SearchResults.jsx
import React from 'react';
import DocumentCard from './DocumentCard';
import './SearchResults.css';

const SearchResults = ({ results, answer, totalResults }) => {
  if (!results || results.length === 0) {
    return (
      <div className="no-results">
        <p>No results found. Try a different query.</p>
      </div>
    );
  }

  // Categorize results by quality
  const excellent = results.filter(r => r.similarity_score >= 0.8);
  const good = results.filter(r => r.similarity_score >= 0.6 && r.similarity_score < 0.8);
  const moderate = results.filter(r => r.similarity_score >= 0.4 && r.similarity_score < 0.6);
  const low = results.filter(r => r.similarity_score < 0.4);

  return (
    <div className="search-results">
      <div className="answer-section">
        <h2>Answer</h2>
        <div className="answer-box">
          {answer.split('\n').map((line, idx) => (
            <p key={idx}>{line}</p>
          ))}
        </div>
      </div>

      <div className="results-header">
        <h2>Sources ({totalResults} documents)</h2>
        <div className="quality-badges">
          {excellent.length > 0 && (
            <span className="quality-badge excellent">
              Excellent: {excellent.length}
            </span>
          )}
          {good.length > 0 && (
            <span className="quality-badge good">
              Good: {good.length}
            </span>
          )}
          {moderate.length > 0 && (
            <span className="quality-badge moderate">
              Moderate: {moderate.length}
            </span>
          )}
          {low.length > 0 && (
            <span className="quality-badge low">
              Low: {low.length}
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
  );
};

export default SearchResults;