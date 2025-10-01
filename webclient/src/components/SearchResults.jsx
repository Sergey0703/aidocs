// src/components/SearchResults.jsx
import React from 'react';
import DocumentCard from './DocumentCard';
import './SearchResults.css';

const SearchResults = ({ results, answer, totalResults }) => {
  // Show answer even if no results
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
              {/* quality badges code... */}
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