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

st.markdown("""
### Input Requirements
- **For 101-feature input**: First 101 values must be > 0 and in range (1e-7, 1e-2)
- **For 103-feature input**: First 101 values must follow same constraints, last 2 values are [c2_value, x_bj_target]
""")

# File uploader for .pkl files
uploaded_file = st.file_uploader("Upload a .pkl file containing the array", type=["pkl"])

# Input fields for c2_value and x_bj_target (used only if needed)
c2_value = st.number_input("C2 Value", value=2.5, min_value=0.0)
x_bj_target = st.number_input("x_Bj Target", value=1e-3, min_value=1e-7, max_value=1e-2, format="%e")

X_input = None
validation_passed = False

if uploaded_file:
    # Load and validate the input
    try:
        file_bytes = BytesIO(uploaded_file.read())
        X_input = pickle.load(file_bytes)
        
        # Ensure it's a 2D array
        if X_input.ndim == 1:
            X_input = X_input.reshape(1, -1)
        
        # Validate input shape and values
        if X_input.shape[1] == 101:
            # Check physical parameters range
            if np.any(X_input <= 0) or np.any(X_input > 1e-2):
                st.error("All 101 features must be > 0 and ≤ 1e-2")
            else:
                # Append c2_value and x_bj_target
                X_input = np.hstack([X_input, np.array([[c2_value, x_bj_target]])])
                validation_passed = True
                
        elif X_input.shape[1] == 103:
            # Check first 101 features
            if np.any(X_input[:, :101] <= 0) or np.any(X_input[:, :101] > 1e-2):
                st.error("First 101 features must be > 0 and ≤ 1e-2")
            # Check x_bj_target (last column)
            elif np.any(X_input[:, -1] <= 1e-7) or np.any(X_input[:, -1] > 1e-2):
                st.error("x_Bj Target (last column) must be in (1e-7, 1e-2]")
            else:
                validation_passed = True
        else:
            st.error(f"Invalid shape: {X_input.shape}. Expected (N, 101) or (N, 103)")
            
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Predict button - only enabled if validation passed
if validation_passed and st.button("Predict"):
    try:
        prediction = rf_model.predict(X_input)
        st.success("Prediction successful!")
        
        # Show first 5 prediction values
        st.markdown(f"**First 5 predicted values:** {prediction[0][:5].round(6)}")
        
        # Download predictions
        pred_bytes = BytesIO()
        pickle.dump(prediction, pred_bytes)
        st.download_button(
            label="Download Predictions (.pkl)",
            data=pred_bytes.getvalue(),
            file_name="predictions.pkl",
            mime="application/octet-stream"
        )

        # Visualization
        st.subheader("Prediction Visualization")
        r_grid = rf_model.Rgrid()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(r_grid, prediction[0], s=10, alpha=0.7)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Radial Distance (log scale)")
        ax.set_ylabel("Amplitude (log scale)")
        ax.set_title("Dipole Amplitude vs Radial Distance")
        ax.grid(True, which="both", ls="--")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Add sample data generation
with st.expander("Need sample data?"):
    st.code("""
import numpy as np
import pickle

# Generate valid sample data (101 features)
X_sample = np.random.uniform(low=1e-7, high=1e-2, size=(5, 101))
with open('sample_input.pkl', 'wb') as f:
    pickle.dump(X_sample, f)
    """)
