import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the cleaned dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


# Create a heatmap showing correlation between all numeric columns
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)

# Add a title to explain what the chart shows
plt.title('Correlation Heatmap')

# Show the final heatmap
plt.show()
