// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import Navigation from './components/common/Navigation';
import SearchPage from './pages/SearchPage';
import IndexingPage from './pages/IndexingPage';

function App() {
  return (
    <Router>
      <div className="app">
        <header className="app-header">
          <h1 className="app-title">Production RAG System</h1>
          <p className="app-subtitle">
            Hybrid Search • Multi-Strategy Intelligence • Smart AI Re-Ranking • Powered by Gemini API
          </p>
          <Navigation />
        </header>

        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/indexing" element={<IndexingPage />} />
        </Routes>

        <footer className="app-footer">
          <p>Production RAG System • Powered by LlamaIndex, Gemini API & React</p>
          <p>Hybrid Search • Database + Vector • Smart AI Re-Ranking</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;