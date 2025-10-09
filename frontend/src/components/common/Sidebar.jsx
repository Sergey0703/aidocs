// src/components/common/Sidebar.jsx
import React from 'react';
import { NavLink } from 'react-router-dom';
import './Sidebar.css';
import { 
  FiSearch, 
  FiUploadCloud, 
  FiGrid, 
  FiInbox // <-- ИМПОРТ НОВОЙ ИКОНКИ
} from 'react-icons/fi';

const Sidebar = () => {
  const menuItems = [
    { to: "/", text: "Search", icon: <FiSearch /> },
    { to: "/indexing", text: "Indexing", icon: <FiUploadCloud /> },
    { to: "/manager", text: "Document Manager", icon: <FiInbox /> }, // <-- НОВАЯ ССЫЛКА
    { to: "/vehicles", text: "Vehicles", icon: <FiGrid /> },
    // Будущие пункты меню
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
            end={item.to === "/"} // 'end' prop для главной страницы
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