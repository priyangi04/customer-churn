import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Plot the number of customers who churned vs who stayed

#create a bar chart using seaborn for churn count
sns.countplot(x='Churn', data=df)
#set the title for the chart
plt.title('Customer Churn Count') 
#renaming the x-axis labls for betterunderstanding
plt.xticks([0, 1], ['Stayed', 'Left'])
# Show the plot
plt.show()

#churn by contrat type
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type')
plt.show()