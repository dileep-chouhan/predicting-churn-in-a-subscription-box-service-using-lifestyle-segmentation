import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Number of customers
n_customers = 500
# Generate synthetic customer data
data = {
    'Age': np.random.randint(18, 65, n_customers),
    'Income': np.random.randint(20000, 150000, n_customers),
    'Subscription_Length': np.random.randint(1, 25, n_customers), # in months
    'Spending_Score': np.random.randint(1, 101, n_customers), # 1-100 scale
    'Churn': np.random.choice([0, 1], size=n_customers, p=[0.8, 0.2]) # 0: No churn, 1: Churn
}
df = pd.DataFrame(data)
# --- 2. Data Analysis and Feature Engineering ---
# Create lifestyle segments based on income and spending score.  This is a simplification.
df['Lifestyle_Segment'] = pd.cut(df['Income'], bins=[0, 50000, 100000, float('inf')], labels=['Budget', 'Mid-Range', 'Luxury'])
# Analyze churn rate by lifestyle segment
churn_by_segment = df.groupby('Lifestyle_Segment')['Churn'].mean()
print("Churn rate by Lifestyle Segment:\n", churn_by_segment)
# --- 3. Predictive Modeling (Simplified) ---
# Prepare data for a simple churn prediction model.
X = df[['Age', 'Income', 'Subscription_Length', 'Spending_Score']]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a RandomForestClassifier (a simple model for demonstration)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# --- 4. Visualization ---
plt.figure(figsize=(8, 6))
sns.barplot(x=churn_by_segment.index, y=churn_by_segment.values)
plt.title('Churn Rate by Lifestyle Segment')
plt.xlabel('Lifestyle Segment')
plt.ylabel('Churn Rate')
plt.savefig('churn_by_segment.png')
print("Plot saved to churn_by_segment.png")
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
print("Plot saved to confusion_matrix.png")