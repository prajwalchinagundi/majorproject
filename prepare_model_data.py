import pandas as pd

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

# Select features
X_train = train_data[selected_features]
X_test = test_data[selected_features]

# Select target
y_train = train_data["label"]
y_test = test_data["label"]

print("Model Data Preparation Completed Successfully!")

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Labels:")
print(y_train.value_counts())

print("\nTesting Labels:")
print(y_test.value_counts())