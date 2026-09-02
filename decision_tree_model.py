import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load processed datasets
train_data = pd.read_csv("dataset/train_processed.csv")
test_data = pd.read_csv("dataset/test_processed.csv")

# Selected features from Step 28
selected_features = [
    "service",
    "flag",
    "wrong_fragment",
    "logged_in",
    "count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

# Prepare training data
X_train = train_data[selected_features]
y_train = train_data["label"]

# Prepare testing data
X_test = test_data[selected_features]
y_test = test_data["label"]

# Create Decision Tree model
dt_model = DecisionTreeClassifier(
    random_state=42
)

# Train the model
print("Training Decision Tree Model...")
dt_model.fit(X_train, y_train)

print("Decision Tree Training Completed Successfully!")

# Make predictions
y_pred = dt_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Accuracy:")
print(f"{accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))