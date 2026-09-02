import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load datasets
train_data = pd.read_csv("dataset/train_processed.csv")
test_data = pd.read_csv("dataset/test_processed.csv")

# Selected features
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

# Prepare data
X_train = train_data[selected_features]
y_train = train_data["label"]

X_test = test_data[selected_features]
y_test = test_data["label"]

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)

print("Training Decision Tree...")
dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

# Display results
print("\n========== MODEL COMPARISON ==========")

print(f"Random Forest Accuracy : {rf_accuracy * 100:.2f}%")
print(f"Decision Tree Accuracy : {dt_accuracy * 100:.2f}%")

# Find best model
if rf_accuracy > dt_accuracy:
    print("\nBest Model: Random Forest")
else:
    print("\nBest Model: Decision Tree")