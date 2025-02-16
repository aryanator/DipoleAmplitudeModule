import streamlit as st
import numpy as np
import pickle
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from DipoleAmplitudePredictor import RandomForestModel

# Initialize the model
rf_model = RandomForestModel()

# Streamlit UI
st.title("Dipole Amplitude Predictor")

st.markdown("### Input the feature array (101 or 103 values)")

# File uploader for .pkl files
uploaded_file = st.file_uploader("Upload a .pkl file containing the array", type=["pkl"])

# Input fields for c2_value and x_bj_target (used only if needed)
c2_value = st.number_input("C2 Value", value=2.5)
x_bj_target = st.number_input("x_Bj Target", value=1e-3)

X_input = None

if uploaded_file:
    # Load .pkl file containing the NumPy array using BytesIO
    try:
        file_bytes = BytesIO(uploaded_file.read())  # Convert the uploaded file into a file-like object
        X_input = pickle.load(file_bytes)  # Load the object from the file-like object
        
        # Ensure it's a 2D array
        if X_input.ndim == 1:
            X_input = X_input.reshape(1, -1)
        
        # Validate input shape (don't raise error for 101, just append c2_value and x_bj_target)
        if X_input.shape[1] == 101:
            # Append c2_value and x_bj_target if input has only 101 features
            X_input = np.hstack([X_input, np.array([[c2_value, x_bj_target]])])
        elif X_input.shape[1] not in [101, 103]:
            st.error(f"Invalid input shape: {X_input.shape}. Expected (N, 101) or (N, 103).")
            st.stop()

    except Exception as e:
        st.error(f"Error loading the .pkl file: {e}")
        st.stop()

# Predict button
if X_input is not None and st.button("Predict"):
    prediction = rf_model.predict(X_input)
    st.success(f"Predicted Output: {prediction}")

    # Convert predictions into a downloadable .pkl file
    try:
        pred_filename = "predictions.pkl"
        with open(pred_filename, "wb") as f:
            pickle.dump(prediction, f)
        st.download_button(
            label="Download All Predictions (.pkl)",
            data=open(pred_filename, "rb").read(),
            file_name=pred_filename,
            mime="application/octet-stream"
        )
    except Exception as e:
        st.error(f"Error saving the predictions: {e}")

    # Visualize only the first prediction
    st.subheader("Prediction Visualization (first prediction)")

    r_grid = rf_model.Rgrid()  # Assuming this gives you the corresponding r_grid array

    # Check if r_grid and prediction have the same length
    if len(r_grid) == len(prediction[0]):
        # Create a log-log plot for the first prediction
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=np.log(r_grid), y=np.log(prediction[0].flatten()))
        plt.title("Log-Scale Visualization of Prediction 1 vs. Rgrid")
        plt.xlabel("Log(Rgrid)")
        plt.ylabel("Log(Predictions)")
        plt.grid(True)
        st.pyplot(plt)  # Display plot in Streamlit
    else:
        st.error("Mismatch between the length of r_grid and predictions.")
