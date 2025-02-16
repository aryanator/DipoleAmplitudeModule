
# Dipole  

A simplified module for easy and quick installation.  

## Installation  

Install the module using the following command:  

```bash
pip install git+https://github.com/aryanator/DipoleAmplitudeModule.git
```  

## Usage  

Once installed, you can initialize and use the model as follows:  

```python
from DipoleAmplitudePredictor import RandomForestModel #import the trained model

rf_model = RandomForestModel()
```

Refer to the file named OldVersion_readme.txt to understand the requirements of passing input to the model
Also, can refer to run.py to see a sample usage of the module

Further, you can also run a webapp by using the follwing command
```bash
streamlit run streamlit_app.py
```

Files:
1. setup.py- Required to setup the module
2. streamlit_app.py- Run the webapp
3. X_new.pkl- pickle file of a sample input numpy arrays for the webpage or the application
4. predictions.pkl- predictions made on X_new

## Features  

- 📦 Easy installation with a single command  
- ⚡ Quick setup and initialization  
- 🛠️ Ready-to-use model for your applications  

## Contributing  

Feel free to open issues or submit pull requests if you have improvements or feature requests.  



⭐ Star this repo if you find it useful! 🚀  
