import numpy as np
import pandas as pd
from pathlib import Path


RANDOM_SEED = 42
OUTPUT_PATH = Path(__file__).resolve().parent / "credit_scoring_data.csv"


def make_dataset(n_samples: int = 5000, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 68, size=n_samples)
    income = rng.lognormal(mean=11.1, sigma=0.45, size=n_samples).astype(float)
    income = np.clip(income, 20000, 250000)

    debt_to_income = rng.beta(2.5, 4.5, size=n_samples) * 0.8 + 0.08
    credit_utilization = rng.beta(2.2, 4.8, size=n_samples) * 0.8 + 0.05
    late_payments = rng.poisson(1.25, size=n_samples)
    credit_history_years = rng.integers(1, 18, size=n_samples)
    loan_amount = rng.lognormal(mean=9.8, sigma=0.7, size=n_samples)
    loan_amount = np.clip(loan_amount, 2000, 150000)
    months_employed = rng.integers(3, 180, size=n_samples)
    savings_balance = rng.lognormal(mean=9.5, sigma=0.85, size=n_samples)
    savings_balance = np.clip(savings_balance, 500, 150000)
    existing_loans = rng.poisson(1.8, size=n_samples)
    monthly_debt_payment = rng.gamma(2.8, 150, size=n_samples)
    employment_status = rng.choice(["Full-time", "Part-time", "Self-employed", "Unemployed"], size=n_samples, p=[0.52, 0.18, 0.19, 0.11])

    risk_score = (
        1.2 * (debt_to_income * 100)
        + 1.5 * (credit_utilization * 100)
        + 7.0 * late_payments
        + 0.8 * np.maximum(existing_loans - 1, 0)
        + 0.5 * np.maximum(24 - credit_history_years, 0)
        + 0.03 * np.maximum(0.0, 120000 - income) / 1000
        + 0.05 * np.maximum(0.0, 60000 - savings_balance) / 1000
        - 0.18 * months_employed / 12
        - 0.4 * np.minimum(1, np.log1p(income) / 12)
    )

    risk_probability = 1 / (1 + np.exp(-(risk_score - 55)))
    risk_flag = rng.binomial(1, risk_probability)

    data = pd.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "debt_to_income_ratio": debt_to_income.round(4),
            "credit_utilization": credit_utilization.round(4),
            "late_payments": late_payments,
            "credit_history_years": credit_history_years,
            "loan_amount": loan_amount.round(2),
            "months_employed": months_employed,
            "savings_balance": savings_balance.round(2),
            "existing_loans": existing_loans,
            "monthly_debt_payment": monthly_debt_payment.round(2),
            "employment_status": employment_status,
            "risk_flag": risk_flag,
        }
    )

    data["credit_score"] = 850 - (data["debt_to_income_ratio"] * 300) - (data["credit_utilization"] * 250) - (data["late_payments"] * 25) + (data["months_employed"] / 12) * 10
    data["credit_score"] = np.clip(data["credit_score"], 300, 850).round(2)

    return data


if __name__ == "__main__":
    dataset = make_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved synthetic credit dataset to {OUTPUT_PATH} ({len(dataset)} rows)")
