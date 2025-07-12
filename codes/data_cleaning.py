import pandas as pd

# Load the data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Peek at the first 5 rows
print("\nFirst 5 rows of data:")
print(df.head())

# Display the shape of the DataFrame
print("\nShape of data (rows, columns):")
print(df.shape)

# Display the columns of the DataFrame
print("\nColumn names:")
print(df.columns)

# Display the data types of the columns
print("\nColumn data types:")
print(df.dtypes)

#converting 'TotalCharges' to numeric, coercing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Display the first 5 rows after conversion
print("\nFirst 5 rows after converting 'TotalCharges' to numeric:")
print(df.head())

# Display the data type of 'TotalCharges'
print("\nData type of 'TotalCharges':")
print(df['TotalCharges'].dtypes)

#Display the number of missing values in each column
print("\nNumber of missing values in each column:")
print(df.isnull().sum())

# Display the number of missing values in 'TotalCharges'
print("\nNumber of missing values in 'TotalCharges':")
print(df['TotalCharges'].isnull().sum())

#remove rows with missing values in 'TotalCharges'
df = df.dropna(subset=['TotalCharges'])

# Display the shape of the DataFrame after removing missing values
print (df.shape)
# Display the first 5 rows after removing missing values
print("\nFirst 5 rows after removing missing values:")
print(df.head())

#cleaning up the DataFrame by removing unnecessary columns
df = df.drop(columns=['customerID'])
# Display the first 5 rows after dropping 'customerID'
print("\nFirst 5 rows after dropping 'customerID':")
print(df.head())
# Display the shape of the DataFrame after dropping 'customerID'
print("\nShape of data after dropping 'customerID':")
print(df.shape)

# Convert binary columns from 'Yes'/'No' to 1/0

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

#conerting gender column to binary. Female=1, Male=0
df['gender'] = df['gender'].map({'Female':1,'Male': 0})

#checking the first 5 rows after converting binary columns and data types
print(df.dtypes)
print("\nFirst 5 rows after converting binary columns:")
print(df.head())

# Churn rate in %
churn_rate = df['Churn'].value_counts(normalize=True) * 100

# Print the % of churned vs not churned
print(churn_rate)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot the number of customers who churned vs who stayed

# Create a bar chart using seaborn
sns.countplot(x='Churn', data=df)

# Save the cleaned data to a new CSV file
df.to_csv("cleaned_customer_churn.csv", index=False)

