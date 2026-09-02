import pandas as pd
import joblib

# Load saved model
model = joblib.load("model/decision_tree_model.pkl")

# Load processed test data
test_data = pd.read_csv("dataset/test_processed.csv")

# Get EXACT features used when the model was trained
model_features = list(model.feature_names_in_)

# Select one real record
sample = test_data.iloc[[0]]

# Actual label
actual_label = sample["label"].iloc[0]

# Use exact model features
X_sample = sample[model_features]

# Prediction
prediction = model.predict(X_sample)[0]

print("========== REAL DATA TEST ==========")

print("\nActual Label:")
print(actual_label)

print("\nPredicted Label:")
print(prediction)

print("\nResult:")

if actual_label == prediction:
    print("CORRECT PREDICTION")
else:
    print("INCORRECT PREDICTION")

print("\n========== 41 FEATURE VALUES ==========")

for feature in model_features:
    print(f"{feature}: {sample[feature].iloc[0]}")