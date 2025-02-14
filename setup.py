from setuptools import setup, find_packages

setup(
    name='DipoleAmplitudePredictor',
    version='0.2',  # Increment version to reflect changes
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'boto3',  # Added new dependencies
    ],
    description='A package for making predictions for Dipole Amplitude using a pre-trained Random Forest model',
    author='Aryan Patil',
    author_email='aryansanjay.patil@stonybrook.edu',
)
