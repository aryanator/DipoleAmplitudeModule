from setuptools import setup, find_packages

setup(
    name='DipoleAmplitudePredictor',
    version='0.2',  # Increment version to reflect changes
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn==1.2.2',  # Specify the exact version
        'boto3',
    ],
    description='A package for making predictions for Dipole Amplitude using a pre-trained Random Forest model',
    author='Aryan Patil',
    author_email='aryansanjay.patil@stonybrook.edu',
    url='https://github.com/aryanator/Dipole/'
)
