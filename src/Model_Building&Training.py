import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, auc
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Load merged dataset
data = pd.read_csv('Data/merged_fraud_data.csv')

# Separate features and target
X = data.drop(['class', 'signup_time', 'purchase_time', 'ip_address', 'user_id', 'device_id', 'ip_int'], axis=1)
y = data['class']



# One-Hot Encoding
categorical_cols = ['source', 'browser', 'sex', 'country']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['purchase_value', 'age', 'time_since_signup', 'user_txn_count', 'device_txn_count']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_balanced.value_counts())

# Train
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_balanced, y_train_balanced)

# Predict
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Evaluate
print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_lr))

# Train
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)

# Predict
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluate
print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_rf))

def evaluate_model(y_true, y_pred, y_proba, name):
    print(f"\n--- {name} ---")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_true, y_proba):.4f}")
    print(f"AUC-PR (Precision-Recall): {average_precision_score(y_true, y_proba):.4f}")

evaluate_model(y_test, y_pred_lr, y_pred_proba_lr, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, y_pred_proba_rf, "Random Forest")

def plot_pr_curve(y_true, y_proba, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_pr = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (AUC-PR = {auc_pr:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_pr_curve(y_test, y_pred_proba_rf, "Random Forest")

# Save best model
os.mkdir("models") if not os.path.exists("models") else None
joblib.dump(rf_model, 'models/random_forest_fraud_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')