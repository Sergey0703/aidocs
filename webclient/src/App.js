// src/App.js
import React, { useState } from 'react';
import './App.css';
import { ragApi } from './api/ragApi';
import SearchBar from './components/SearchBar';
import SystemStatus from './components/SystemStatus';
import SearchResults from './components/SearchResults';

function App() {
  const [searchResults, setSearchResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (query, maxResults, rerankMode) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await ragApi.search(query, maxResults, rerankMode);
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
          Hybrid Search • Multi-Strategy Intelligence • Smart AI Re-Ranking • Powered by Gemini API
        </p>
      </header>

      <div className="app-container">
        <aside className="sidebar">
          <SystemStatus lastSearchMetrics={searchResults?.performance_metrics} />
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
              <SearchResults 
                results={searchResults.results}
                answer={searchResults.answer}
                totalResults={searchResults.total_results}
                entityResult={searchResults.entity_result}
                rewriteResult={searchResults.rewrite_result}
                performanceMetrics={searchResults.performance_metrics}
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
                <li>🧠 Smart entity extraction with AI</li>
                <li>🔄 Intelligent query rewriting</li>
                <li>🗄️ Database exact matching</li>
                <li>🔍 Vector semantic search</li>
                <li>⚡ Smart AI Re-Ranking (auto-skip when not needed)</li>
                <li>🎯 Advanced results fusion</li>
              </ul>
              <div className="rerank-explainer">
                <h3>🤖 AI Re-Ranking Modes:</h3>
                <div className="mode-card smart">
                  <div className="mode-header">
                    <span className="mode-icon">🧠</span>
                    <span className="mode-name">Smart Mode (Default)</span>
                  </div>
                  <p>Automatically skips AI verification for exact database matches, saving ~4 seconds and 2,500 tokens per query. Used in ~70% of searches.</p>
                </div>
                <div className="mode-card full">
                  <div className="mode-header">
                    <span className="mode-icon">🚀</span>
                    <span className="mode-name">Full Mode</span>
                  </div>
                  <p>Always uses AI to verify ALL documents for maximum accuracy (90-95%). Best for critical queries where accuracy matters most.</p>
                </div>
              </div>
              <p className="example-query">
                Try: "tell me about John Nolan" or "AI automation"
              </p>
            </div>
          )}
        </main>
      </div>

      <footer className="app-footer">
        <p>Production RAG System • Powered by LlamaIndex, Gemini API & React</p>
        <p>Hybrid Search • Database + Vector • Smart AI Re-Ranking</p>
      </footer>
    </div>
  );
}

export default App;