// src/api/ragApi.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const ragApi = {
  // Search endpoint
  search: async (query, maxResults = 20, rerankMode = 'smart') => {
    const response = await api.post('/api/search', {
      query,
      max_results: maxResults,
      rerank_mode: rerankMode,  // ⭐ Добавлен параметр
    });
    return response.data;
  },

  // System status
  getStatus: async () => {
    const response = await api.get('/api/system/status');
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await api.get('/api/system/health');
    return response.data;
  },
};