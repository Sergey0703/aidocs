// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import SearchPage from './pages/SearchPage';
import IndexingPage from './pages/IndexingPage';
import VehiclesPage from './pages/VehiclesPage'; // Создадим эту страницу позже
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<SearchPage />} />
          <Route path="indexing" element={<IndexingPage />} />
          <Route path="vehicles" element={<VehiclesPage />} />
          {/* 
            Здесь будут будущие роуты:
            <Route path="buildings" element={<BuildingsPage />} />
            <Route path="reports" element={<ReportsPage />} />
            <Route path="settings" element={<SettingsPage />} />
          */}
        </Route>
      </Routes>
    </Router>
  );
}

export default App;