import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load processed datasets
train_data = pd.read_csv("dataset/train_processed.csv")
test_data = pd.read_csv("dataset/test_processed.csv")

# Separate features and labels
X_train = train_data.drop("label", axis=1)
y_train = train_data["label"]

X_test = test_data.drop("label", axis=1)
y_test = test_data["label"]

print("Training Decision Tree model...")

# Create Decision Tree model
model = DecisionTreeClassifier(
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("Decision Tree training completed!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Accuracy:")
print(accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model/decision_tree_model.pkl")

print("\nDecision Tree model saved successfully!")