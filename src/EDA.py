# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# SKLearn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Set visual style
sns.set(style='whitegrid')
plt.style.use('ggplot')

# Define paths
data_path = './Data/'

# --------------------------
# 1. Load datasets
# --------------------------
fraud_data = pd.read_csv(data_path + 'Fraud_Data.csv')
ip_country = pd.read_csv(data_path + 'IpAddress_to_Country.csv')
credit_card = pd.read_csv(data_path + 'creditcard.csv')

print("=== Fraud Data Sample ===")
print(fraud_data.head())

print("\n=== IP to Country Sample ===")
print(ip_country.head())

print("\n=== Credit Card Data Sample ===")
print(credit_card.head())

# --------------------------
# 2. Data Cleaning
# --------------------------
print("\n=== Missing Values ===")
print("Fraud Data:", fraud_data.isnull().sum().sum())
print("Credit Card:", credit_card.isnull().sum().sum())
print("IP Country:", ip_country.isnull().sum().sum())

# Drop missing values and duplicates
fraud_data.dropna(inplace=True)
credit_card.dropna(inplace=True)
ip_country.dropna(inplace=True)

fraud_data.drop_duplicates(inplace=True)
credit_card.drop_duplicates(inplace=True)
ip_country.drop_duplicates(inplace=True)

# Convert timestamps
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])

print("\n=== Fraud Data Types ===")
print(fraud_data.dtypes)

# --------------------------
# 3. Class Distribution (Fraud vs Non-Fraud)
# --------------------------
plt.figure(figsize=(8, 4))
sns.countplot(x='class', data=fraud_data)
plt.title('E-commerce: Fraud vs Non-Fraud')
plt.xlabel('Class (0 = Non-Fraud, 1 = Fraud)')
plt.ylabel('Count')
plt.show()

print("\n=== E-commerce Class Distribution ===")
print(fraud_data['class'].value_counts(normalize=True))

plt.figure(figsize=(8, 4))
sns.countplot(x='Class', data=credit_card)
plt.title('Bank: Fraud vs Non-Fraud')
plt.xlabel('Class (0 = Non-Fraud, 1 = Fraud)')
plt.ylabel('Count')
plt.show()

print("\n=== Bank Class Distribution ===")
print(credit_card['Class'].value_counts(normalize=True))

# --------------------------
# 4. EDA - Numerical & Categorical Features
# --------------------------
plt.figure(figsize=(10, 4))
sns.histplot(fraud_data['purchase_value'], bins=50, kde=True)
plt.title('E-commerce Purchase Value Distribution')
plt.xlabel('Purchase Value ($)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(credit_card['Amount'], bins=50, kde=True)
plt.title('Bank Transaction Amount Distribution')
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(fraud_data['purchase_value'], bins=50, log_scale=True)
plt.title('E-commerce Purchase Value (Log Scale)')
plt.xlabel('Purchase Value ($)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(credit_card['Amount'], bins=50, log_scale=True)
plt.title('Bank Transaction Amount (Log Scale)')
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(y='source', data=fraud_data, order=fraud_data['source'].value_counts().index)
plt.title('E-commerce: Traffic Source')
plt.xlabel('Count')
plt.ylabel('Source')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(y='browser', data=fraud_data, order=fraud_data['browser'].value_counts().index)
plt.title('E-commerce: Browser Usage')
plt.xlabel('Count')
plt.ylabel('Browser')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='sex', data=fraud_data)
plt.title('E-commerce: Gender Distribution')
plt.xlabel('Gender (M/F)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(fraud_data['age'], bins=20, kde=True)
plt.title('E-commerce: Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# --------------------------
# 5. Feature Engineering
# --------------------------
# Time-based features
fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600  # in hours
fraud_data['purchase_hour'] = fraud_data['purchase_time'].dt.hour

# Transaction frequency (per user and device)
fraud_data['user_txn_count'] = fraud_data.groupby('user_id')['user_id'].transform('count')
fraud_data['device_txn_count'] = fraud_data.groupby('device_id')['device_id'].transform('count')

# IP to Country Merge
fraud_data['ip_int'] = fraud_data['ip_address'].astype(int)
ip_country['lower_bound_ip_address'] = ip_country['lower_bound_ip_address'].astype(int)
ip_country['upper_bound_ip_address'] = ip_country['upper_bound_ip_address'].astype(int)

ip_country_sorted = ip_country.sort_values('lower_bound_ip_address')
fraud_data_sorted = fraud_data.sort_values('ip_int')

merged_data = pd.merge_asof(fraud_data_sorted, ip_country_sorted,
                            left_on='ip_int',
                            right_on='lower_bound_ip_address')

# Filter valid IP range
merged_data = merged_data[
    (merged_data['ip_int'] >= merged_data['lower_bound_ip_address']) &
    (merged_data['ip_int'] <= merged_data['upper_bound_ip_address'])
]

print("\n=== Merged Data Sample (IP + Country) ===")
print(merged_data[['ip_address', 'country', 'purchase_value', 'class']].head())

# --------------------------
# 6. Data Transformation
# --------------------------
# Define features and target
X = merged_data.drop(['class', 'signup_time', 'purchase_time', 'ip_address', 'user_id', 'device_id', 'ip_int'], axis=1, errors='ignore')
y = merged_data['class']

# One-Hot Encoding
categorical_cols = ['source', 'browser', 'sex', 'country']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Scale numerical features
numerical_cols = ['purchase_value', 'age', 'time_since_signup', 'user_txn_count', 'device_txn_count']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])  # Use transform, not fit

# --------------------------
# 7. Handle Class Imbalance
# --------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE:", y_res.value_counts())

# Optional: SMOTE + Undersampling
over = SMOTE(sampling_strategy=0.2)
under = RandomUnderSampler(sampling_strategy=0.5)
pipeline = Pipeline(steps=[('over', over), ('under', under)])
X_res, y_res = pipeline.fit_resample(X_train, y_train)
print("After SMOTE + Undersampling:", y_res.value_counts())

# --------------------------
# 8. Model Training & Evaluation
# --------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_res, y_res)

y_pred = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print("ROC AUC Score:", roc_auc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Fraud Detection")
plt.show()

# --------------------------
# 9. Save Cleaned Datasets
# --------------------------
merged_data.to_csv('Data/merged_fraud_data.csv', index=False)
credit_card.to_csv('Data/cleaned_creditcard.csv', index=False)

# --------------------------
# 10. Final Summary
# --------------------------
print("\n=== E-commerce Data Summary ===")
print(merged_data.describe(include='all'))

print("\n=== Bank Data Summary ===")
print(credit_card.describe(include='all'))