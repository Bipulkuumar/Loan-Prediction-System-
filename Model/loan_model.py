import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("dataset/loan_data.csv")

# Separate input features and output
X = data[['cibil', 'income', 'employment', 'loan_amount',
          'tenure', 'avg_balance', 'emi', 'transactions']]
y = data['result']

# Split data into training and testing (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100, "%")

# Save trained model
pickle.dump(model, open("model/loan_approval_model.pkl", "wb"))
