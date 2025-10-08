// src/components/common/Sidebar.jsx
import React from 'react';
import { NavLink } from 'react-router-dom';
import './Sidebar.css';
import { 
  FiSearch, 
  FiUploadCloud, 
  FiGrid, 
  FiBarChart2, 
  FiSettings 
} from 'react-icons/fi'; // Импортируем иконки

const Sidebar = () => {
  const menuItems = [
    { to: "/", text: "Search", icon: <FiSearch /> },
    { to: "/indexing", text: "Indexing", icon: <FiUploadCloud /> },
    { to: "/vehicles", text: "Vehicles", icon: <FiGrid /> },
    // Будущие пункты меню
    // { to: "/buildings", text: "Buildings", icon: <FiHome /> },
    // { to: "/reports", text: "Reports", icon: <FiBarChart2 /> },
    // { to: "/settings", text: "Settings", icon: <FiSettings /> },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2 className="sidebar-title">I-ADMS</h2>
      </div>
      <nav className="sidebar-nav">
        {menuItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
            end // 'end' prop ensures only exact route match is active for "/"
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-text">{item.text}</span>
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-footer">
        <p>v1.1.0</p>
      </div>
    </aside>
  );
};

export default Sidebar;