// src/components/common/Navigation.jsx
import React from 'react';
import { NavLink } from 'react-router-dom';
import './Navigation.css';

const Navigation = () => {
  return (
    <nav className="navigation">
      <div className="nav-container">
        <NavLink 
          to="/" 
          className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
        >
          <span className="nav-icon">🔍</span>
          <span className="nav-text">Search</span>
        </NavLink>
        
        <NavLink 
          to="/indexing" 
          className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
        >
          <span className="nav-icon">📄</span>
          <span className="nav-text">Document Indexing</span>
        </NavLink>
      </div>
    </nav>
  );
};

export default Navigation;