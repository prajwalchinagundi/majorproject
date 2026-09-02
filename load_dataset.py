import pandas as pd

# NSL-KDD column names
columns = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label"
]

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

# Training data has an additional difficulty-level column.
# We do not need it for intrusion prediction.
if train_data.shape[1] == 43:
    train_data = train_data.iloc[:, :42]

# Assign column names
train_data.columns = columns
test_data.columns = columns

print("NSL-KDD Dataset Loaded Successfully")

print("\nTraining Dataset:")
print("Rows:", train_data.shape[0])
print("Columns:", train_data.shape[1])

print("\nTesting Dataset:")
print("Rows:", test_data.shape[0])
print("Columns:", test_data.shape[1])

print("\nTraining Dataset Columns:")
print(train_data.columns.tolist())

print("\nAttack Labels:")
print(train_data["label"].value_counts().head(10))