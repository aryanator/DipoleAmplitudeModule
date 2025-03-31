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

# Improved range visualization
st.markdown("""
<style>
.range-indicator {
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
</style>

<div class="range-indicator">
    <strong>⚠️ x_Bj Target Range Constraint:</strong><br>
    1×10⁻⁷ < x ≤ 1×10⁻² (0.0000001 to 0.01)
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload .pkl file (101 or 103 features)", type=["pkl"])

# Improved x_Bj target input
col1, col2 = st.columns(2)
with col1:
    x_bj_target = st.slider(
        "x_Bj Target Value (scientific notation)",
        min_value=1e-7,
        max_value=1e-2,
        value=1e-3,
        step=1e-7,
        format="%.1e",
        help="Slide or use ←→ arrows to adjust"
    )
with col2:
    c2_value = st.number_input(
        "C2 Value", 
        value=2.5,
        step=0.1,
        format="%.1f"
    )

# Current value display
st.markdown(f"""
<div class="range-indicator">
    <strong>Current x_Bj Target:</strong> {x_bj_target:.2e} ({x_bj_target:.8f})
</div>
""", unsafe_allow_html=True)

# Validation and prediction
if uploaded_file:
    try:
        X_input = pickle.load(BytesIO(uploaded_file.read()))
        
        # Shape handling
        if X_input.ndim == 1:
            X_input = X_input.reshape(1, -1)
            
        # x_Bj validation
        if X_input.shape[1] == 101:
            # Use UI input for x_Bj (already validated)
            X_input = np.hstack([X_input, [[c2_value, x_bj_target]]])
            st.success("✅ Using validated x_Bj target from slider input")
            
        elif X_input.shape[1] == 103:
            # Check last value is x_Bj target
            x_bj_uploaded = X_input[:, -1]
            if np.any((x_bj_uploaded <= 1e-7) | (x_bj_uploaded > 1e-2)):
                st.error("""
                ❌ Invalid x_Bj in uploaded file!
                Last value must satisfy: 1×10⁻⁷ < x ≤ 1×10⁻²
                """)
                st.stop()
            st.success("✅ Valid x_Bj target detected in uploaded array")
            
        else:
            st.error("Invalid shape: Must be (N, 101) or (N, 103)")
            st.stop()

        # Prediction
        if st.button("Run Prediction"):
            with st.spinner('Calculating...'):
                prediction = rf_model.predict(X_input)
                
                # Visualization
                st.subheader("Prediction Results")
                r_grid = rf_model.Rgrid()
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.loglog(r_grid, prediction[0], 'o-', markersize=4)
                ax.set_xlabel("Radial Distance (log scale)")
                ax.set_ylabel("Amplitude (log scale)")
                ax.grid(True, which="both", linestyle='--', alpha=0.5)
                st.pyplot(fig)
                
                # Numerical results
                st.download_button(
                    label="Download Predictions",
                    data=pickle.dumps(prediction),
                    file_name="dipole_predictions.pkl",
                    mime="application/octet-stream"
                )
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
