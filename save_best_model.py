import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

# Load processed training data
train_data = pd.read_csv("dataset/train_processed.csv")

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

# Prepare training data
X_train = train_data[selected_features]
y_train = train_data["label"]

# Create the best model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
print("Training Best Model...")
dt_model.fit(X_train, y_train)

# Save the model
joblib.dump(dt_model, "decision_tree_model.pkl")

# Save selected feature names
joblib.dump(selected_features, "selected_features.pkl")

print("Best Model Saved Successfully!")
print("Model file: decision_tree_model.pkl")
print("Features file: selected_features.pkl")