# Import necessary libraries for data and visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the cleaned dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


# Create a boxplot to compare Monthly Charges of people who stayed vs left
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)

# Add a title to explain what the chart shows
plt.title('Monthly Charges vs Churn')

# Rename the x-axis labels: 0 = Stayed, 1 = Left
plt.xticks([0, 1], ['Stayed', 'Left'])

# Show the final plot
plt.show()



# Create a histogram showing how long customers stayed, split by churn
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30)

# Add a title to explain what this chart shows
plt.title('Tenure Distribution by Churn')

# Label the x-axis to explain what tenure means
plt.xlabel('Tenure (Months)')

# Label the y-axis to show number of customers
plt.ylabel('Customer Count')

# Show the final plot
plt.show()

