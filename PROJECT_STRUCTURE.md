# Veritas-AI Project Structure

This document outlines the proposed scalable directory structure for the Veritas-AI project. This design ensures clear separation between Machine Learning pipelines, backend API logic, and future expansion areas.

## 🌳 Directory Tree

```text
Veritas-AI/
├── .github/                      # CI/CD Workflows (GitHub Actions)
├── backend/                      # Python FastAPI logic
│   ├── app/                      # Main application source code
│   │   ├── api/                  # API Route controllers
│   │   │   ├── v1/               # API versioning
│   │   │   │   ├── endpoints/    # Individual route files
│   │   │   │   │   ├── predict.py     # Feature 1: Fake News Prediction
│   │   │   │   │   ├── explain.py     # Future: Explainable AI logic
│   │   │   │   │   └── audit.py       # Future: Metadata/Image Audit
│   │   │   │   └── api.py        # Central API router
│   │   ├── core/                 # Global configuration and security
│   │   │   ├── config.py         # Env vars, project metadata
│   │   │   └── security.py       # JWT, authentication, CORS
│   │   ├── db/                   # Database session and connections
│   │   │   ├── base.py           # SQLAlchemy declarative base
│   │   │   └── session.py        # Session dependency injection
│   │   ├── models/               # Database ORM models (SQLAlchemy)
│   │   │   ├── news_history.py   # Logs for past predictions
│   │   │   └── user.py           # User accounts
│   │   ├── schemas/              # Pydantic models (Input/Output validation)
│   │   │   ├── request.py        # Incoming prediction payloads
│   │   │   └── response.py       # Outgoing JSON structures
│   │   ├── services/             # Core logic & ML serving
│   │   │   ├── classifier.py     # LR Model inference service
│   │   │   └── explainer.py      # SHAP/LIME logic
│   │   ├── main.py               # FastAPI entry point
│   │   └── tests/                # Unit & Integration tests (pytest)
│   ├── scripts/                  # ML Development Pipeline
│   │   ├── data_cleaning.py      # Preprocessing logic
│   │   └── model_training.py     # Logic to train and save .pkl files
│   ├── ml_models/                # Serialized model artifacts (.pkl, .onnx)
│   │   ├── model_v1.pkl          # Current Logistic Regression model
│   │   └── tfidf_vectorizer.pkl  # Vectorizer artifact
│   ├── datasets/                 # Local data storage
│   │   ├── raw/                  # Original CSV/JSON imports
│   │   └── processed/            # Final data used for training
│   ├── requirements.txt          # Backend dependencies
│   └── .env                      # Secret environment variables
├── frontend/                     # Placeholder for UI (React/Next.js/Streamlit)
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── ...
├── docs/                         # Project wikis and API documentation
├── docker-compose.yml            # Multi-container orchestration
├── .gitignore                    # Excluded files (venv, .env, __pycache__)
└── README.md                     # High-level project overview
```

---

## 📂 Major Directory Breakdown

| Directory | Purpose |
| :--- | :--- |
| **`backend/app/api`** | Handles HTTP routing. Versioned (v1/) to ensure future additions (v2/) don't break existing clients. |
| **`backend/app/services`** | Where the "heavy lifting" happens. It abstracts the ML model inference away from the API routes. |
| **`backend/app/schemas`** | Defines the contract for data. Validates incoming requests and outgoing responses. |
| **`backend/app/models`** | Defines the database schema for persistent storage (e.g., saving predicted results). |
| **`backend/ml_models`** | Version-controlled storage for trained models and vectorizers. |
| **`backend/scripts`** | Home of the ML development pipeline (cleaning, training, evaluation). |
| **`backend/datasets`** | Organized storage for raw input data and processed training sets. |
| **`frontend/`** | Isolated directory for the UI, allowing modular frontend development. |
| **`core/`** | Centralized management for environment variables and security settings. |
