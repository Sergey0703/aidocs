// src/components/SearchResults.jsx
import React from 'react';
import DocumentCard from './DocumentCard';
import './SearchResults.css';

const SearchResults = ({ results, answer, totalResults }) => {
  // Show answer if it exists
  if (answer) {
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

        {results && results.length > 0 && (
          <>
            <div className="results-header">
              <h2>Sources ({totalResults} documents)</h2>
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
          </>
        )}
      </div>
    );
  }

  // No answer and no results
  return (
    <div className="no-results">
      <p>No results found. Try a different query.</p>
    </div>
  );
};

export default SearchResults;