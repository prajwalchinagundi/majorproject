import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. Load processed testing data
# ==========================================

test_data = pd.read_csv("dataset/test_processed.csv")


# ==========================================
# 2. Separate features and target
# ==========================================

X_test = test_data.drop("label", axis=1)
y_test = test_data["label"]


# ==========================================
# 3. Convert test labels to numbers
#    normal = 0
#    attack = 1
# ==========================================

y_test = y_test.map({
    "normal": 0,
    "attack": 1
})


# Check for invalid labels
if y_test.isnull().any():
    print("Error: Unknown labels found in test data.")
    print(test_data["label"].unique())
    exit()


# Convert to NumPy array
y_test = y_test.astype(int).to_numpy()


# ==========================================
# 4. Load trained models
# ==========================================

random_forest = joblib.load(
    "model/random_forest_model.pkl"
)

decision_tree = joblib.load(
    "model/decision_tree_model.pkl"
)


# ==========================================
# 5. Make predictions
# ==========================================

rf_pred = random_forest.predict(X_test)

dt_pred = decision_tree.predict(X_test)
# ==========================================
# Confusion Matrix
# ==========================================

rf_cm = confusion_matrix(y_test, rf_pred)

dt_cm = confusion_matrix(y_test, dt_pred)

print("\n================ RANDOM FOREST CONFUSION MATRIX ================\n")
print(rf_cm)

print("\n================ DECISION TREE CONFUSION MATRIX ================\n")
print(dt_cm)


# ==========================================
# 6. Convert model predictions to numbers
#    normal = 0
#    attack = 1
# ==========================================

label_mapping = {
    "normal": 0,
    "attack": 1
}


rf_pred = pd.Series(rf_pred).map(label_mapping)
dt_pred = pd.Series(dt_pred).map(label_mapping)


# Check for invalid predictions
if rf_pred.isnull().any():
    print("Error: Random Forest produced unknown labels.")
    print(np.unique(random_forest.predict(X_test)))
    exit()


if dt_pred.isnull().any():
    print("Error: Decision Tree produced unknown labels.")
    print(np.unique(decision_tree.predict(X_test)))
    exit()


# Convert predictions to NumPy arrays
rf_pred = rf_pred.astype(int).to_numpy()
dt_pred = dt_pred.astype(int).to_numpy()


# ==========================================
# 7. Calculate metrics
# ==========================================

def calculate_metrics(name, y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "Model": name,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1-Score": f1 * 100
    }


# ==========================================
# 8. Calculate results
# ==========================================

results = []


results.append(
    calculate_metrics(
        "Random Forest",
        y_test,
        rf_pred
    )
)


results.append(
    calculate_metrics(
        "Decision Tree",
        y_test,
        dt_pred
    )
)


# ==========================================
# 9. Create comparison table
# ==========================================

comparison = pd.DataFrame(results)


print("\n================ MODEL COMPARISON ================\n")

print(
    comparison.to_string(index=False)
)


# ==========================================
# 10. Save comparison results
# ==========================================

comparison.to_csv(
    "model_comparison.csv",
    index=False
)


print("\n===================================================")
print("Model comparison saved successfully!")
print("File: model_comparison.csv")
print("===================================================")