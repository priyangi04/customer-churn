# Import basic libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("cleaned_customer_churn.csv")


# First, define your target variable — what you're trying to predict
# Here, we want to predict whether a customer will churn (1) or not (0)
target = 'Churn'

# Now define the features (the input columns the model will use to make predictions)
# We'll exclude 'Churn' and any text-based or ID columns we don't need
features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'
]

# Optional: print the shape to confirm what we’re working with
print("Number of features:", len(features))

# we import the train_test_split function from scikit-learn
from sklearn.model_selection import train_test_split

# Splitting the data into input (X) and target/output (y)
X = df[features]   # Features we defined earlier
y = df[target]     # The 'Churn' column

# Now splitting X and y into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing
# random_state=42 ensures we get the same split every time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Printing the shapes to confirm
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# Now importing Logistic Regression from sklearn
from sklearn.linear_model import LogisticRegression

# Creating an instance of the Logistic Regression model
# 'max_iter' is set high so it doesn't stop too early
model = LogisticRegression(max_iter=1000)

# Fitting the model (train it) using the training data
model.fit(X_train, y_train)

# Print the model’s accuracy on training data (just for info)
train_accuracy = model.score(X_train, y_train)
print("Training Accuracy:", round(train_accuracy * 100, 2), "%")


# Add prediction results to the original test set
X_test['Actual_Churn'] = y_test
X_test['Predicted_Churn'] = y_train

# Optional: add back customer IDs or merge with original df if needed
# Save to CSV
X_test.to_csv('churn_predictions.csv', index=False)
