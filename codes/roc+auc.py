# -------------------------------------
# STEP 1: Import all required libraries
# -------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# -------------------------------------
# STEP 2: Load and clean the dataset
# -------------------------------------

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Replace blank strings with actual NaN values
df.replace(" ", pd.NA, inplace=True)

# Convert TotalCharges to numeric (invalid blanks become NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Map Yes/No to 1/0 for specific columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Map gender to numeric
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

# Drop any rows with missing values
df = df.dropna()

# Confirm shape after cleaning
print("✅ Shape after cleaning:", df.shape)

# -------------------------------------
# STEP 3: Select features and target
# -------------------------------------

features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'
]

X = df[features]
y = df['Churn']

# -------------------------------------
# STEP 4: Train-test split
# -------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# -------------------------------------
# STEP 5: Train the model
# -------------------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------------
# STEP 6: Predict and evaluate
# -------------------------------------

y_pred = model.predict(X_test)

print("\n✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------------
# STEP 7: Plot ROC Curve and AUC Score
# -------------------------------------

# Get prediction probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Get FPR and TPR
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Get AUC Score
auc_score = roc_auc_score(y_test, y_probs)
print("\n🔥 AUC Score:", round(auc_score, 4))

# Plot ROC
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='purple', label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], color='pink', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
