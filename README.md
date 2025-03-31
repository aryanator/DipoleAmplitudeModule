# Dipole Amplitude Predictor

## 📦 Installation
```bash
# PyPI (stable release)
pip install DipoleAmplitudePredictor==0.2.1

# GitHub (latest development version)
pip install git+https://github.com/aryanator/DipoleAmplitudeModule.git
```

## 🔧 Data Requirements
### For Python Module Usage (Direct)
- **Strictly 103 features** in this order:
  ```python
  [101 physical parameters] + [c2_value] + [x_bj_target]
  ```
  - Shape: `(N, 103)` where N = number of samples
  - Example:
    ```python
    import numpy as np
    X = np.hstack([physical_params, [[2.5, 0.001]]])  # Append c2 and x_bj
    ```

### For Web App Usage
- **Accepts either format**:
  - `(N, 101)`: Physical parameters only *(app auto-appends your UI inputs for c2/x_bj)*
  - `(N, 103)`: Full features (same as Python API)

## 🐍 Python Module Usage
```python
from DipoleAmplitudePredictor import RandomForestModel
import numpy as np

# Sample 103-feature input
physical_params = np.random.rand(1, 101)  # Your 101 core features
c2, x_bj = 2.5, 1e-3
X = np.hstack([physical_params, [[c2, x_bj]]])  # -> (1, 103)

# Predict
model = RandomForestModel()
predictions = model.predict(X)  # Returns (N, len_r_grid) array
r_grid = model.Rgrid()  # Corresponding radial grid points
```

## 🌐 Web App Guide
🔗 Live Demo: [https://dipole-amplitude-prediction.streamlit.app/](https://dipole-amplitude-prediction.streamlit.app/)

### How to Use:
1. **Upload** `.pkl` file containing:
   - *(Option 1)*: `(N, 101)` array (physical params only)
   - *(Option 2)*: `(N, 103)` array (full features)
2. **Manually input** `c2_value` and `x_bj_target` if using 101-feature input
3. **Click Predict** to:
   - Get numerical predictions
   - Visualize log-scale amplitude vs. radial distance
   - Download results as `.pkl`

### Sample Files:
- [X_new_101.pkl](sample_link): 101-feature example
- [X_new_103.pkl](sample_link): Complete 103-feature example

## 🚀 Key Features
- **Smart Input Handling**: Web app auto-completes partial (101-feature) inputs
- **Physics-Ready**: Maintains strict 103-feature requirement for Python API
- **Validation**: Both interfaces check input shapes rigorously

---
