import pandas as pd

# Load training dataset
train_data = pd.read_csv(
    "dataset/KDDTrain+.txt",
    sep="\t",
    header=None
)

# Load testing dataset
test_data = pd.read_csv(
    "dataset/KDDTest+.txt",
    sep="\t",
    header=None
)

print("Original Training Shape:", train_data.shape)
print("Original Testing Shape:", test_data.shape)

# Column names for training dataset
train_columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

# Column names for testing dataset
test_columns = train_columns[:-1]

train_data.columns = train_columns
test_data.columns = test_columns

# Convert categorical columns to numbers
categorical_columns = [
    "protocol_type",
    "service",
    "flag"
]

for column in categorical_columns:

    # Combine training and testing values
    combined = pd.concat([
        train_data[column].astype(str),
        test_data[column].astype(str)
    ])

    # Create mapping
    unique_values = combined.unique()

    mapping = {
        value: index
        for index, value in enumerate(unique_values)
    }

    # Apply mapping
    train_data[column] = train_data[column].astype(str).map(mapping)
    test_data[column] = test_data[column].astype(str).map(mapping)

# Convert attack labels into two classes
train_data["label"] = train_data["label"].astype(str).str.strip().str.rstrip(".")
test_data["label"] = test_data["label"].astype(str).str.strip().str.rstrip(".")

train_data["label"] = train_data["label"].apply(
    lambda x: "normal" if x.lower() == "normal" else "attack"
)

test_data["label"] = test_data["label"].apply(
    lambda x: "normal" if x.lower() == "normal" else "attack"
)
# Remove difficulty column from training data
train_data = train_data.drop(columns=["difficulty"])

# Save processed datasets
train_data.to_csv(
    "dataset/train_processed.csv",
    index=False
)

test_data.to_csv(
    "dataset/test_processed.csv",
    index=False
)

print("\nPreprocessing completed successfully!")

print("\nTraining Dataset:")
print("Rows:", train_data.shape[0])
print("Columns:", train_data.shape[1])

print("\nTesting Dataset:")
print("Rows:", test_data.shape[0])
print("Columns:", test_data.shape[1])

print("\nTraining Labels:")
print(train_data["label"].value_counts())

print("\nTesting Labels:")
print(test_data["label"].value_counts())

print("\nProcessed datasets saved successfully!")