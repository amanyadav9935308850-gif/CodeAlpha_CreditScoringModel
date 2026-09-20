# Credit Scoring Model

This project builds a machine learning-based credit scoring model to predict whether an applicant is likely to be a low-risk or high-risk borrower using historical financial indicators.

## Objective

Predict an individual's creditworthiness using past financial data.

## Approach

- Feature engineering from financial history
- Training and comparison of classification models
- Model evaluation using metrics such as Precision, Recall, F1-score, and ROC-AUC

## Dataset

The project generates a synthetic credit dataset with realistic financial variables such as income, debt-to-income ratio, credit utilization, payment history, employment length, and loan behavior.

## Project structure

- `data/generate_dataset.py` — generates a synthetic dataset
- `notebooks/credit_scoring_notebook.ipynb` — interactive Jupyter version of the project
- `src/train_model.py` — trains and evaluates the model
- `src/inference.py` — predicts risk for a single applicant using the saved model
- `models/` — stores trained artifacts

## Setup

```bash
cd credit_scoring_model
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the project

For the notebook version:

```bash
python -m notebook notebooks/credit_scoring_notebook.ipynb
```

For the script version:

```bash
python data/generate_dataset.py
python src/train_model.py
python src/inference.py --income 75000 --debt_to_income 0.42 --credit_utilization 0.65 --late_payments 2 --months_employed 42 --loan_amount 18000 --monthly_debt_payment 650 --savings_balance 15000 --existing_loans 2 --credit_history_years 6
```

## Expected output

The training script compares multiple models and saves the best model to `models/best_credit_model.joblib`.

## Files generated

- `data/credit_scoring_data.csv`
- `models/best_credit_model.joblib`
- `models/metrics.json`
