import pandas as pd
import joblib

# Load saved model
model = joblib.load("model/decision_tree_model.pkl")

print("\nMODEL LOADED SUCCESSFULLY")

# Get features used by the model
model_features = list(model.feature_names_in_)

print("\n========== MODEL FEATURES ==========")
print(model_features)
print("Number of model features:", len(model_features))

# Load test dataset
test_data = pd.read_csv("dataset/test_processed.csv")

print("\n========== TEST DATA ==========")
print("Test data shape:", test_data.shape)
print("Test data columns:")
print(list(test_data.columns))

# Check which model features are missing
missing = [f for f in model_features if f not in test_data.columns]

print("\n========== MISSING FEATURES ==========")
print(missing)
print("Number missing:", len(missing))

# Check extra features
extra = [f for f in test_data.columns if f not in model_features and f != "label"]

print("\n========== EXTRA FEATURES ==========")
print(extra)

# Only predict if everything matches
if len(missing) == 0:

    X_test = test_data[model_features]
    y_test = test_data["label"]

    print("\n========== FEATURE CHECK ==========")
    print("Features sent to model:", X_test.shape[1])

    print("\nMaking prediction...")

    y_pred = model.predict(X_test)

    print("Prediction successful!")

else:

    print("\n❌ FEATURE MISMATCH FOUND")
    print("The test dataset does not contain all features required by the model.")
    print("DO NOT run prediction until this is fixed.")