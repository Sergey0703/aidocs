# Advanced RAG Document Q&A System

This project is a sophisticated, production-ready RAG (Retrieval-Augmented Generation) system designed to answer questions based on a private collection of documents. It specializes in extracting information about people, leveraging a powerful hybrid search mechanism that combines vector-based semantic search with direct database keyword search.

The backend is built with Python and designed as an API, making it easy to integrate with any custom frontend. The system uses Google Gemini for LLM tasks and Supabase (PostgreSQL with pgvector) for data storage and retrieval.

## Key Features

-   **🚀 Hybrid Search:** Combines semantic vector search with exact-match database search for superior accuracy and recall.
-   **🧠 Smart Entity Extraction:** Automatically identifies key entities (like people's names) in user queries using advanced NLP techniques.
-   **✏️ Multi-Query Rewriting:** Expands the original user query into several variants to cover different search angles.
-   **⚖️ Advanced Results Fusion:** Intelligently merges and ranks results from multiple retrieval strategies to provide the most relevant answers.
-   **🤖 API-First Design:** Built as a standalone backend API (e.g., using FastAPI/Flask), ready to be consumed by any frontend application (web, mobile, etc.).
-   **✨ Powered by Google Gemini:** Utilizes Google's powerful Gemini models for embeddings and intelligent NLP tasks.
-   **💾 Supabase/PostgreSQL Backend:** Leverages the power and flexibility of SQL and `pgvector` for efficient hybrid storage and retrieval.

## Architecture Overview

The system is designed with a clean separation between the frontend, backend, and data layers:

`Frontend (e.g., Vercel)` ➡️ `Backend API (e.g., AWS, Render)` ➡️ `Supabase DB & Google Gemini API`

This README focuses on setting up and running the **Backend API**.

## Prerequisites

Before you begin, ensure you have the following installed:
-   Python 3.8+
-   Git

## Getting Started: Backend Setup

Follow these steps to set up and run the backend service on your local machine or a server.

### 1. Clone the Repository

First, clone the new repository to your local machine:
```bash
git clone https://github.com/your-username/your-new-repository-name.git
cd your-new-repository-name
```

### 2. Create and Activate a Virtual Environment

It is crucial to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.

**On macOS / Linux:**
```bash
# Create the virtual environment (in a folder named 'venv')
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```
*(You will see `(venv)` at the beginning of your terminal prompt, indicating it's active.)*

**On Windows:**
```bash
# Create the virtual environment (in a folder named 'venv')
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```
*(You will see `(venv)` at the beginning of your terminal prompt, indicating it's active.)*

### 3. Install Dependencies

With your virtual environment activated, install all the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

The application requires API keys and connection strings to be configured in an environment file.

-   Create a copy of `.env.example` (if it exists) or create a new file named `.env` in the project root.
-   Add the following variables to your `.env` file:

```env
# Your Supabase connection string (PostgreSQL format)
SUPABASE_CONNECTION_STRING="postgresql://postgres:[YOUR-PASSWORD]@[YOUR-DB-HOST]:5432/postgres"

# Your Google Gemini API Key
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

# You can also override other settings from config/settings.py here if needed
# For example:
# TABLE_NAME="my_custom_table"
```

## Running the Backend

The backend has two main processes: the **Indexer** (run once to process documents) and the **API Server** (runs continuously to answer queries).

### Step 1: Run the Indexer

Before you can perform searches, you must process your documents and populate the database.

> **Important:** Make sure your `DOCUMENTS_DIR` is correctly set in `config.py` or your `.env` file before running the indexer.

To start the indexing process, run:
```bash
# Make sure your virtual environment is active!
python indexer.py
```
This process may take a long time depending on the number and size of your documents.

### Step 2: Run the API Server

Once the indexing is complete, you can start the API server. This server will expose the endpoints that your frontend application will call.

Assuming you have created an `api.py` file with FastAPI, run the following command:
```bash
# Make sure your virtual environment is active!
uvicorn api:app --reload
```
-   `api`: The name of your Python file (e.g., `api.py`).
-   `app`: The name of the FastAPI instance inside your file.
-   `--reload`: Automatically restarts the server when you make changes to the code (great for development).

Your backend API is now running and ready to accept requests! You can typically access it at `http://127.0.0.1:8000`.

---
Client start:

#streamlit run main_app.py
python run_api.py
# Перейдите в родительскую папку
cd C:\projects\aidocs

# Создайте React проект (это займет несколько минут)
npx create-react-app webclient

# Перейдите в созданный проект
cd webclient

# Установите зависимости
npm install axios react-markdown

npm start