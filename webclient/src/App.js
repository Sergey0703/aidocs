// src/App.js
import React, { useState } from 'react';
import './App.css';
import { ragApi } from './api/ragApi';
import SearchBar from './components/SearchBar';
import SystemStatus from './components/SystemStatus';
import EntityInfo from './components/EntityInfo';
import PerformanceMetrics from './components/PerformanceMetrics';
import SearchResults from './components/SearchResults';

function App() {
  const [searchResults, setSearchResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (query, maxResults) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await ragApi.search(query, maxResults);
      setSearchResults(result);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Search failed');
      console.error('Search error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearResults = () => {
    setSearchResults(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">Production RAG System</h1>
        <p className="app-subtitle">
          Hybrid Search • Multi-Strategy Intelligence • Advanced Fusion • Powered by Gemini API
        </p>
      </header>

      <div className="app-container">
        <aside className="sidebar">
          <SystemStatus />
        </aside>

        <main className="main-content">
          <SearchBar onSearch={handleSearch} isLoading={isLoading} />

          {error && (
            <div className="error-message">
              <h3>Search Error</h3>
              <p>{error}</p>
              <button onClick={handleClearResults} className="clear-button">
                Clear
              </button>
            </div>
          )}

          {isLoading && (
            <div className="loading-container">
              <div className="loading-spinner"></div>
              <p>Executing hybrid search pipeline...</p>
            </div>
          )}

          {searchResults && !isLoading && (
            <>
              <EntityInfo 
                entityResult={searchResults.entity_result}
                rewriteResult={searchResults.rewrite_result}
              />

              <PerformanceMetrics metrics={searchResults.performance_metrics} />

              <SearchResults 
                results={searchResults.results}
                answer={searchResults.answer}
                totalResults={searchResults.total_results}
              />

              <div className="clear-section">
                <button onClick={handleClearResults} className="clear-results-button">
                  Clear Results
                </button>
              </div>
            </>
          )}

          {!searchResults && !isLoading && !error && (
            <div className="welcome-message">
              <h2>Welcome to Production RAG System</h2>
              <p>Enter your question above to search using our hybrid approach combining:</p>
              <ul>
                <li>Entity extraction with AI</li>
                <li>Intelligent query rewriting</li>
                <li>Database exact matching</li>
                <li>Vector semantic search</li>
                <li>Advanced results fusion</li>
              </ul>
              <p className="example-query">
                Try: "tell me about ..."
              </p>
            </div>
          )}
        </main>
      </div>

      <footer className="app-footer">
        <p>Production RAG System • Powered by LlamaIndex, Gemini API & React</p>
        <p>Hybrid Search • Database + Vector • Multi-Strategy Intelligence</p>
      </footer>
    </div>
  );
}

export default App;