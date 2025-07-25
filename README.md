  🚀 Improved Fraud Detection for E-commerce and Bank Transactions  
  By Firaol Delesa | 10 Academy: Artificial Intelligence Mastery    
  Adey Innovations Inc.  


   📌 Project Overview

This project aims to improve fraud detection in   e-commerce and banking transactions   by leveraging   machine learning, geolocation analysis, and transaction pattern recognition  . The goal is to build accurate, explainable models that balance   security   and   user experience  .

Using real-world datasets, we:
- Cleaned and preprocessed transaction data
- Engineered time-based and behavioral features
- Built and compared models (Logistic Regression vs Random Forest)
- Used   SHAP explainability   to interpret model decisions

The final model helps   prevent financial losses   while   building trust   with customers and institutions.

   🎯 Business Context

Fraud detection is a high-stakes challenge:
-   False positives   → alienate legitimate customers
-   False negatives   → lead to direct financial loss

This system enables:
- Real-time fraud monitoring
- Actionable insights for risk teams
- Transparent decision-making via model explainability

   📁 Project Structure

```
Improved_fraud_detection_for_e-commerce_and_bank_transactions/
│
├── Data/
│   ├── Fraud_Data.csv                    E-commerce transaction data
│   ├── IpAddress_to_Country.csv          IP range to country mapping
│   ├── creditcard.csv                    Bank transaction data
│   ├── merged_fraud_data.csv             Cleaned e-commerce data with country
│   └── cleaned_creditcard.csv            Cleaned bank data
│
├── models/
│   ├── random_forest_fraud_model.pkl     Trained Random Forest model
│   └── scaler.pkl                        Fitted StandardScaler
│
├── src/
│   ├── EDA.py                            Data cleaning, EDA, and preprocessing
│   ├── Model_building.py                 Model training and evaluation
│   └── Shap_analysis.py                  SHAP explainability
│
├── reports/
│   ├── shap_summary_plot.png             Global feature importance
│   ├── shap_force_plot.png               Local prediction explanation
│   ├── shap_dependence_ .png             Feature impact analysis
│   └── shap_feature_importance.csv       Top 10 features by SHAP value
│
├── README.md                             This file
├── requirements.txt                      Python dependencies
└── .gitignore
```

   🧠 Key Features Engineered

| Feature | Description |
|-------|-------------|
| `time_since_signup` | Duration (in hours) between signup and purchase |
| `purchase_hour` | Hour of day when purchase occurred |
| `user_txn_count` | Number of transactions per user |
| `device_txn_count` | Number of transactions per device |
| `country` | Geolocation from IP address (via `IpAddress_to_Country.csv`) |

These features help identify suspicious behavior like:
- Immediate purchases after signup
- High transaction velocity
- Use of high-risk geolocations


   ⚖️ Handling Class Imbalance

Both datasets are highly imbalanced:
-   E-commerce  : 90.6% non-fraud, 9.4% fraud
-   Bank  : 99.8% non-fraud, 0.17% fraud

    Strategy:
- Applied   SMOTE + Random Undersampling   on the   training set only  
- Evaluated using   F1-Score, AUC-PR, ROC-AUC   instead of accuracy

> This ensures the model learns to detect rare fraud cases without overfitting.

   🤖 Model Comparison

| Model | F1-Score | AUC-PR | ROC-AUC | Interpretability |
|------|----------|--------|--------|------------------|
|   Logistic Regression   | 0.17 | 0.10 | 0.51 | ✅ High |
|   Random Forest   |   0.66   |   0.71   |   0.85   | ❌ Low (but fixed with SHAP) |

    ✅ Final Model:   Random Forest  

  Why?  
- Significantly higher F1-score and AUC-PR
- Better at capturing complex, non-linear fraud patterns
- When combined with SHAP, becomes   fully explainable  


   🔍 Model Explainability with SHAP

Used   SHAP (SHapley Additive exPlanations)   to interpret the Random Forest model:

    1.   SHAP Summary Plot  
![SHAP Summary Plot](reports/shap_summary_plot.png)  
 Top drivers of fraud: `time_since_signup`, `purchase_value`, `user_txn_count` 

    2.   SHAP Force Plot  
![SHAP Force Plot](reports/shap_force_plot.png)  
 Why a specific transaction was flagged as fraud 

    3.   SHAP Dependence Plots  
- `time_since_signup`: Short durations → high fraud risk
- `purchase_value`: High amounts → higher risk
- `user_txn_count`: Unusual frequency → suspicious


   📊 Key Insights

1.   Short `time_since_signup` is the strongest fraud predictor    
   → Fraudsters often make purchases within minutes of signing up.

2.   High transaction velocity is a red flag    
   → Users or devices with unusually high transaction counts are suspicious.

3.   Geolocation matters    
   → Certain countries show higher fraud rates.

4.   Large purchases are riskier    
   → High `purchase_value` increases fraud likelihood.

   🛠️ How to Run

    1. Clone the Repository
```bash
git clone https://github.com/FiraolD/Improved_fraud_detection_for_e-commerce_and_bank_transactions.git

cd Improved_fraud_detection_for_e-commerce_and_bank_transactions
```

    2. Install Dependencies
```bash
pip install -r requirements.txt
```

    3. Run the Full Pipeline
```bash
  Step 1: Data Analysis & Preprocessing
python src/EDA.py

  Step 2: Model Training
python src/Model_building.py

  Step 3: Model Explainability
python src/Shap_analysis.py
```

   📦 Requirements

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
shap
joblib
```

Install with:
```bash
pip install -r requirements.txt
```

   📅 Key Dates (10 Academy)

| Task | Due Date |
|------|----------|
| Interim-1 Submission | 20 July 2025 |
| Interim-2 Submission | 27 July 2025 |
| Final Submission | 29 July 2025 |

   🚀 Final Note

This project demonstrates a complete   end-to-end fraud detection pipeline   — from raw data to explainable AI. It’s ready for real-world deployment and further scaling.