import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# Load Saved Model and Features
# --------------------------------------------------
model = joblib.load("model/decision_tree_model.pkl")
selected_features = list(model.feature_names_in_)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🛡️ AI-Based Network Intrusion Detection System")
st.subheader("Machine Learning Based Cyber Attack Detection")

st.write(
    "Enter the network traffic feature values below. "
    "The trained Decision Tree model will classify the traffic "
    "as **Normal** or **Attack**."
)

st.divider()

# --------------------------------------------------
# Input Features
# --------------------------------------------------
st.header("Network Traffic Features")

# Create two columns
col1, col2 = st.columns(2)

input_values = {}

for i, feature in enumerate(selected_features):

    if i % 2 == 0:
        with col1:
            input_values[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                step=1.0
            )
    else:
        with col2:
            input_values[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                step=1.0
            )

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
st.divider()

if st.button("🔍 Detect Network Traffic", use_container_width=True):

    # Create DataFrame in correct feature order
    input_data = pd.DataFrame(
        [[input_values[feature] for feature in selected_features]],
        columns=selected_features
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == "attack":
        st.error("🚨 ATTACK DETECTED!")
        st.write("The network traffic has been classified as malicious.")

    elif prediction == "normal":
        st.success("✅ NORMAL TRAFFIC")
        st.write("The network traffic appears to be normal.")

    else:
        st.warning(f"Prediction: {prediction}")

    # Display input data
    with st.expander("View Input Data"):
        st.dataframe(input_data)