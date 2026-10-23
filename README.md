# 📉 Telco Customer Churn Prediction

> End-to-end data mining project that predicts which telecom customers are about to churn — so retention teams can intervene before they leave.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

---

## 📖 Overview

Customer churn is the silent killer of subscription businesses. Acquiring a new customer costs **~5× more** than retaining an existing one, so every churner you save is direct revenue.

This project builds a **binary classification model** that predicts churn risk from customer behavior, contract details, and service usage — and translates model outputs into **business-actionable risk scores**.

### 🎯 Business Question
> *Which customers are most likely to cancel in the next billing cycle, and what drives their decision?*

### ✅ Success Criteria
| Metric | Target | Rationale |
|--------|--------|-----------|
| **Recall (Churn class)** | ≥ 0.75 | Missing a churner costs ~$500; false alarms only cost ~$50 in retention offers |
| **PR-AUC** | ≥ 0.60 | Robust metric for imbalanced data |
| **ROC-AUC** | ≥ 0.83 | Overall discriminative power |

---

## 🗂️ Project Structure
churn-prediction/
│
├── data/
│ ├── raw/ # Original dataset (not committed)
│ └── processed/ # Cleaned & feature-engineered data
│
├── notebooks/
│ ├── 01_eda.ipynb # Exploratory data analysis
│ ├── 02_feature_engineering.ipynb
│ ├── 03_modeling.ipynb # Training & evaluation
│ └── 04_interpretation.ipynb # SHAP analysis
│
├── src/
│ ├── init.py
│ ├── data_loader.py # Data ingestion
│ ├── preprocessing.py # Cleaning, encoding, scaling
│ ├── features.py # Custom feature engineering
│ ├── train.py # Model training pipeline
│ ├── evaluate.py # Metrics & reports
│ └── predict.py # Inference on new data
│
├── models/
│ └── churn_model.pkl # Serialized model artifact
│
├── reports/
│ ├── figures/ # EDA plots, SHAP plots
│ └── model_report.md # Full evaluation write-up
│
├── tests/
│ ├── test_preprocessing.py
│ └── test_predict.py
│
├── requirements.txt
├── setup.py
├── Makefile
└── README.md


---

## 📊 Dataset

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|----------|-------|
| Records | 7,043 customers |
| Features | 20 (numerical + categorical) |
| Target | `Churn` (Yes / No) |
| Class Balance | 73.5% No · 26.5% Yes |
| Missing Values | `TotalCharges` (11 nulls, hidden as blank strings) |

### Key Features
- Demographics: gender, senior citizen, partner, dependents
- Account: tenure, contract type, payment method, paperless billing
- Services: phone, internet, online security, streaming, tech support
- Charges: monthly charges, total charges

> ⚠️ The dataset is **not committed** to this repo. Download it manually into `data/raw/`.

---

🔬 Methodology
Raw Data → Clean → Engineer Features → Encode → Scale
        → Train/Test Split (stratified) → Model Training
        → Cross-Validation → Tuning → Evaluation → SHAP → Deploy

📈 Results
Final Model: Gradient Boosting (tuned)

Test Set Performance (20% holdout):
Metric	Score
ROC-AUC	0.849
PR-AUC	0.641
Precision (Churn)	0.54
Recall (Churn)	0.79
F1 (Churn)	0.64
Accuracy	0.78

Confusion Matrix
                 Predicted No    Predicted Yes
Actual No           903              132
Actual Yes           82              292

Business translation:

    ✅ 292 churners caught → potential savings ~$146,000

    ⚠️ 82 churners missed → lost revenue ~$41,000

    💸 132 false alarms → wasted offers ~$6,600

    Net estimated value: ~$98,000 per scoring cycle

🔍 Top Churn Drivers (SHAP)

https://reports/figures/shap_summary.png

Ranked by impact:

    Contract type — Month-to-month customers churn 15× more than 2-year

    Tenure — Churn risk drops sharply after 12 months

    Internet service — Fiber optic users churn more (price / service pain)

    Payment method — Electronic check payers churn most

    Monthly charges — Higher bills → higher churn

Insight: Retention should focus on new, month-to-month fiber customers paying by e-check — that's the highest-risk segment.

🤝 Contributing

Contributions welcome! Please:

    Fork the repo

    Create a feature branch (git checkout -b feature/xyz)

    Commit your changes (git commit -m 'Add xyz')

    Push and open a PR

📄 License

MIT License — see LICENSE for details.

🙏 Acknowledgments

    Dataset: IBM Sample Data — Kaggle

    Inspiration: CRISP-DM methodology

⭐ If this project helped you, drop a star! ⭐
