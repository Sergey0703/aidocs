// src/components/SearchBar.jsx
import React, { useState } from 'react';
import './SearchBar.css';

const SearchBar = ({ onSearch, isLoading }) => {
  const [query, setQuery] = useState('');
  const [maxResults, setMaxResults] = useState(20);
  const [rerankMode, setRerankMode] = useState('smart'); // 'smart' or 'full'

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query, maxResults, rerankMode);
    }
  };

  return (
    <div className="search-bar-container">
      <form onSubmit={handleSubmit} className="search-form">
        <div className="search-input-group">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your question (e.g., tell me about ...)"
            className="search-input"
            disabled={isLoading}
          />
          <input
            type="number"
            value={maxResults}
            onChange={(e) => setMaxResults(parseInt(e.target.value))}
            min="1"
            max="100"
            className="max-results-input"
            disabled={isLoading}
            title="Max results"
          />
          <button 
            type="submit" 
            className="search-button"
            disabled={isLoading || !query.trim()}
          >
            {isLoading ? 'Searching...' : 'Hybrid Search'}
          </button>
        </div>

        {/* Re-Ranking Mode Selection */}
        <div className="rerank-options">
          <div className="rerank-label">
            <span className="label-text">AI Re-Ranking:</span>
            <span className="label-hint">Choose verification mode</span>
          </div>
          
          <div className="rerank-modes">
            <label className={`rerank-option ${rerankMode === 'smart' ? 'active' : ''}`}>
              <input
                type="radio"
                name="rerankMode"
                value="smart"
                checked={rerankMode === 'smart'}
                onChange={(e) => setRerankMode(e.target.value)}
                disabled={isLoading}
              />
              <span className="option-content">
                <span className="option-icon">🧠</span>
                <span className="option-details">
                  <span className="option-title">Smart</span>
                  <span className="option-description">Auto-skip when not needed (recommended)</span>
                </span>
              </span>
              <span className="option-badge recommended">Default</span>
            </label>

            <label className={`rerank-option ${rerankMode === 'full' ? 'active' : ''}`}>
              <input
                type="radio"
                name="rerankMode"
                value="full"
                checked={rerankMode === 'full'}
                onChange={(e) => setRerankMode(e.target.value)}
                disabled={isLoading}
              />
              <span className="option-content">
                <span className="option-icon">🚀</span>
                <span className="option-details">
                  <span className="option-title">Full</span>
                  <span className="option-description">Always verify with AI (slower, max accuracy)</span>
                </span>
              </span>
              <span className="option-badge full">+Accuracy</span>
            </label>
          </div>

          <div className="rerank-info">
            {rerankMode === 'smart' && (
              <div className="info-box smart">
                <span className="info-icon">💡</span>
                <span className="info-text">
                  Smart mode automatically skips AI verification for exact database matches, 
                  saving time and tokens (~70% of queries).
                </span>
              </div>
            )}
            {rerankMode === 'full' && (
              <div className="info-box full">
                <span className="info-icon">⚠️</span>
                <span className="info-text">
                  Full mode uses AI to verify ALL documents for maximum accuracy. 
                  This is slower and uses more tokens (recommended for critical queries).
                </span>
              </div>
            )}
          </div>
        </div>
      </form>
    </div>
  );
};

export default SearchBar;