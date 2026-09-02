import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# Load processed training data
train_data = pd.read_csv("dataset/train_processed.csv")

# Separate features and target
X = train_data.drop("label", axis=1)
y = train_data["label"]

# Select top 20 important features
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]

print("Feature Selection Completed Successfully!")
print("\nSelected Features:")
for feature in selected_features:
    print(feature)

print("\nNumber of Selected Features:", len(selected_features))