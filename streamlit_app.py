import streamlit as st
import numpy as np
import pickle
from io import BytesIO
import matplotlib.pyplot as plt
from DipoleAmplitudePredictor import RandomForestModel

# Initialize model
rf_model = RandomForestModel()

# Streamlit UI
st.title("Dipole Amplitude Predictor")

# File uploader
uploaded_file = st.file_uploader("Upload .pkl file (101 or 103 features)", type=["pkl"])

# Input fields with clear constraints
c2_value = st.number_input("C2 Value", value=2.5)

x_bj_target = st.number_input(
    "x_Bj Target (MUST be between 1e-7 and 1e-2)", 
    value=1e-3,
    min_value=1e-7,
    max_value=1e-2,
    format="%e",
    help="Value will be automatically clamped to valid range"
)

# Validation and prediction
if uploaded_file:
    try:
        X_input = pickle.load(BytesIO(uploaded_file.read()))
        
        # Shape handling
        if X_input.ndim == 1:
            X_input = X_input.reshape(1, -1)
            
        # x_Bj validation
        if X_input.shape[1] == 101:
            # Use UI input for x_Bj (already range-constrained by Streamlit)
            X_input = np.hstack([X_input, [[c2_value, x_bj_target]]])
            
        elif X_input.shape[1] == 103:
            # Check last value is x_Bj target
            x_bj_uploaded = X_input[:, -1]
            if np.any((x_bj_uploaded <= 1e-7) | (x_bj_uploaded > 1e-2)):
                st.error("ERROR: Last value must be 1e-7 < x_Bj ≤ 1e-2")
                st.stop()
            
        else:
            st.error("Invalid shape: Must be (N, 101) or (N, 103)")
            st.stop()

        # Prediction
        if st.button("Predict"):
            prediction = rf_model.predict(X_input)
            
            # Visualization
            r_grid = rf_model.Rgrid()
            fig, ax = plt.subplots()
            ax.loglog(r_grid, prediction[0], 'o-')
            ax.set_xlabel("Radial Distance")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
