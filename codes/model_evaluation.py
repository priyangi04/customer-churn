# -----------------------------
# STEP 1: Import all libraries
# -----------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ----------------------------------
# STEP 2: Load and clean the dataset
# ----------------------------------

# Load the Telco Churn dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert 'gender' to numeric: Female = 1, Male = 0
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

# Convert 'TotalCharges' to numeric (invalid blanks become NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Convert Yes/No columns to 1/0 for model compatibility
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Drop rows that still have missing values
df = df.dropna()

# Check shape after cleaning
print("New shape after cleaning:", df.shape)

# -------------------------------------------------------
# STEP 3: Define features (X) and target variable (y)
# -------------------------------------------------------

# Select input features
features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'
]

# Define X and y
X = df[features]
y = df['Churn']

# --------------------------------------
# STEP 4: Split the data into train/test
# --------------------------------------

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print shapes to verify
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# ---------------------------------------------
# STEP 5: Train the model (Logistic Regression)
# ---------------------------------------------

# Create and fit the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ------------------------------------------
# STEP 6: Predict and Evaluate the Model
# ------------------------------------------

# Predict churn on test set
y_pred = model.predict(X_test)

# Print evaluation metrics
print("\nTest Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


from sklearn.metrics import roc_curve, roc_auc_score

# Get predicted probabilities for class 1 (churn = 1)
y_probs = model.predict_proba(X_test)[:, 1]

# Calculate FPR, TPR for different thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calculate AUC Score
auc_score = roc_auc_score(y_test, y_probs)
print("\nAUC Score:", round(auc_score, 4))

# Plot the ROC curve
plt.plot(fpr, tpr, color='navy', label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
