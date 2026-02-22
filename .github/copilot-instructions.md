# Copilot Instructions

This project is currently in its initial stages. It is a Python project utilizing a virtual environment (`.venv`).

## Project Context
- **Language**: Python
- **Environment**: Virtual environment (`.venv`) is used for dependency management.
- **Architecture**: The proposed architecture has two parallel processes:
  1. **Time-Series Model**: Facebook Prophet to model the non-linear nature of historical price data (e.g., NIFTY 50 using `yfinance`).
  2. **Semantic Analyzer Model**: FinBERT (`yiyanghkust/finbert-tone`) to extract sentiment signals from financial news (e.g., Indian Financial News Dataset).
- **Integration**: The models will be combined and integrated with XAI (Explainable AI) principles using Shapley Additive Explanations (SHAP) for higher interpretability.
- **Domain**: XAI, Deep Learning, NLP, and Financial Forecasting.

## Developer Workflows
- **Dependencies**: Ensure the virtual environment is activated before installing dependencies or running scripts.
  ```bash
  source .venv/bin/activate
  ```
- **Adding Code**: As new modules and packages are added, ensure they follow standard Python conventions (PEP 8).
- **Data**: Data files are stored in the `data/` directory.
- **App**: Application code is stored in the `app/` directory.

## AI Agent Guidelines
- **Code Generation**: Write clean, idiomatic Python code.
- **Documentation**: Include docstrings for all new functions, classes, and modules.
- **Testing**: As the project grows, ensure tests are written alongside new features.

*(Note: This file should be updated as the project architecture, specific frameworks, and conventions are established.)*
