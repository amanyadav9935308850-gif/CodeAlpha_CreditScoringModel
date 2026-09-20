import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_credit_model.joblib"


def build_payload(args):
    return {
        "age": args.age,
        "income": args.income,
        "debt_to_income_ratio": args.debt_to_income,
        "credit_utilization": args.credit_utilization,
        "late_payments": args.late_payments,
        "credit_history_years": args.credit_history_years,
        "loan_amount": args.loan_amount,
        "months_employed": args.months_employed,
        "savings_balance": args.savings_balance,
        "existing_loans": args.existing_loans,
        "monthly_debt_payment": args.monthly_debt_payment,
        "employment_status": args.employment_status,
        "credit_score": args.credit_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict credit risk for a single applicant.")
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--income", type=float, required=True)
    parser.add_argument("--debt_to_income", type=float, required=True)
    parser.add_argument("--credit_utilization", type=float, required=True)
    parser.add_argument("--late_payments", type=int, required=True)
    parser.add_argument("--credit_history_years", type=int, required=True)
    parser.add_argument("--loan_amount", type=float, required=True)
    parser.add_argument("--months_employed", type=int, required=True)
    parser.add_argument("--savings_balance", type=float, required=True)
    parser.add_argument("--existing_loans", type=int, required=True)
    parser.add_argument("--monthly_debt_payment", type=float, required=True)
    parser.add_argument("--employment_status", type=str, required=True, choices=["Full-time", "Part-time", "Self-employed", "Unemployed"])
    parser.add_argument("--credit_score", type=float, required=True)
    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)
    payload = pd.DataFrame([build_payload(args)])
    prediction = model.predict(payload)[0]
    probability = model.predict_proba(payload)[0, 1]

    label = "High risk" if prediction == 1 else "Low risk"
    print(json.dumps({
        "prediction": int(prediction),
        "label": label,
        "risk_probability": round(float(probability), 4),
    }, indent=2))


if __name__ == "__main__":
    main()
