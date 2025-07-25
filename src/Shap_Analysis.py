# shap_analysis.py
# Task 3: Model Explainability with SHAP
# Improved Fraud Detection for E-commerce and Bank Transactions
# By Firaol Delesa | Adey Innovations Inc.

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)

# Create reports folder
import os
os.makedirs('reports', exist_ok=True)

print("🚀 Starting SHAP Analysis for Task 3...")

# --------------------------
# 1. Load Model and Scaler
# --------------------------
print("\n📦 Loading model and scaler...")
try:
    model = joblib.load('models/random_forest_fraud_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Model and scaler loaded.")
except Exception as e:
    raise FileNotFoundError(f"❌ Load failed: {e}")

# --------------------------
# 2. Load and Preprocess Data
# --------------------------
print("\n📊 Loading and preprocessing data...")
try:
    data = pd.read_csv('Data/merged_fraud_data.csv')
    print(f"✅ Loaded data: {data.shape}")
except FileNotFoundError:
    raise FileNotFoundError("❌ merged_fraud_data.csv not found. Run EDA.py first.")

# Prepare features
X = data.drop(['class', 'signup_time', 'purchase_time', 'ip_address', 'user_id', 'device_id', 'ip_int'], axis=1)
y = data['class']

# One-Hot Encode
categorical_cols = ['source', 'browser', 'sex', 'country']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Scale numerical features
numerical_cols = ['purchase_value', 'age', 'time_since_signup', 'user_txn_count', 'device_txn_count']
X[numerical_cols] = scaler.transform(X[numerical_cols])

# Ensure float64
X = X.astype(np.float64)
print(f"✅ Final feature matrix: {X.shape}")


# --------------------------
# 3. Sample and Convert to NumPy
# --------------------------
print("\n🔧 Preparing SHAP sample...")
X_sample_df = shap.sample(X, 100)  # Use 100 for speed
X_sample = X_sample_df.values.astype(np.float64)  # Critical: Convert and cast
feature_names = X_sample_df.columns.tolist()
print(f"✅ SHAP sample shape: {X_sample.shape}")


# --------------------------
# 4. Create Explainer with Masker
# --------------------------
print("\n🧠 Creating SHAP explainer...")

# Wrap predict_proba to ensure clean output
def predict_proba_wrapper(x):
    if isinstance(x, np.ndarray):
        x_df = pd.DataFrame(x, columns=feature_names)
    else:
        x_df = x
    return model.predict_proba(x_df)

# Use Independent masker for stability
masker = shap.maskers.Independent(data=X_sample)

explainer = shap.Explainer(
    predict_proba_wrapper,
    masker=masker,
    output_names=['Non-Fraud', 'Fraud']
)

# Compute SHAP values
shap_values = explainer(X_sample)
print("✅ SHAP values computed!")

# --------------------------
# 5. Summary Plot
# --------------------------
print("\n📈 Generating Summary Plot...")
#shap.summary_plot()(shap_values[:, :, 1], show=False)
shap.summary_plot(shap_values[:, :, 1], show=False)
plt.title("SHAP Summary Plot: Key Drivers of Fraud", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('reports/shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Summary plot saved.")

# --------------------------
# 6. SHAP Force Plot (Local Explanation)
# --------------------------
print("\n🔍 Generating SHAP Force Plot for a Fraud Case...")

try:
    # Use the first sample
    sample_idx = 0
    shap_value_for_sample = shap_values[sample_idx]

    # Get base value (expected value for class 1)
    base_value = shap_value_for_sample.base_values[1]
    
    # Get SHAP values for fraud class (class 1)
    shap_vals = shap_value_for_sample.values[:, 1]

    # Create force plot
    shap.plots.force(
        base_value,
        shap_vals,
        X_sample_df.iloc[sample_idx],
        matplotlib=True,
        show=False
    )
    plt.title("SHAP Force Plot: Why This Transaction Was Flagged as Fraud")
    plt.tight_layout()
    plt.savefig('reports/shap_force_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ SHAP Force Plot saved to 'reports/shap_force_plot.png'")
except Exception as e:
    print(f"❌ Error generating force plot: {e}")


# --------------------------
# 7. Dependence Plots
# --------------------------
print("\n📉 Generating Dependence Plots...")
features = ['time_since_signup', 'purchase_value', 'user_txn_count']
for feature in features:
    if feature in feature_names:
        idx = feature_names.index(feature)
        shap.plots.scatter(shap_values[:, idx, 1], show=False)
        plt.title(f"SHAP Dependence: {feature}")
        plt.tight_layout()
        plt.savefig(f'reports/shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Dependence plot for '{feature}' saved.")


# --------------------------
# 8. Feature Importance
# --------------------------
print("\n🏆 Top 10 Features by SHAP Importance:")
shap_df = pd.DataFrame(shap_values.values[:, :, 1], columns=feature_names)
importance = shap_df.abs().mean().sort_values(ascending=False).head(10)
print(importance.to_string())
importance.to_csv('reports/shap_feature_importance.csv')
print("✅ Feature importance saved.")


# --------------------------
# 9. Completion
# --------------------------
print("\n" + "="*60)
print("           ✅ SHAP ANALYSIS COMPLETED")
print("="*60)
print("✅ All plots and data saved in 'reports/'")
print("📄 Ready for final report submission!")



























