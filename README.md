# CO2_prediction_BiLSTM

**Project:** Surface seawater CO₂ prediction (μatm) using a Bidirectional LSTM  
**Author:** Data Analysis Team  
**Date:** September 2025

## Summary
Jupyter notebook implementing a Bidirectional LSTM (BiLSTM) to predict surface seawater CO₂ concentrations (μatm) from multivariate time series. The workflow includes data ingestion and cleaning, preprocessing and windowing, GroupKFold cross-validation, feature importance via permutation, and exporting predictions.

## Goals
- Prepare and window multivariate time series for LSTM modeling.
- Train a robust BiLSTM with group-aware cross-validation.
- Evaluate performance and estimate feature importance.
- Export predictions and artifacts for downstream analysis.

## Notebook structure
1. Data loading and initial exploration  
2. Cleaning and preprocessing (normalization, imputation)  
3. Building time windows (sliding windows)  
4. Defining and training the BiLSTM model  
5. Validation with GroupKFold and metrics (MAE, RMSE, R²)  
6. Feature importance (permutation-based)  
7. Saving and exporting predictions

## Quick requirements
- Python 3.8+  
- Key dependencies: numpy, pandas, scikit-learn, tensorflow/keras, matplotlib, seaborn

Example install:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

## Recommended usage
- Run cells sequentially in a reproducible environment (virtualenv/conda).  
- Adjust data paths and window/model parameters in the configuration section.  
- Fix random seeds to improve reproducibility for training and splitting.

## Best practices
- Ensure GroupKFold groups correctly capture temporal/spatial dependencies.  
- Log hyperparameters and metrics (e.g., MLflow or CSV exports).  
- Check for seasonal biases before interpreting feature importance.

---

# Installation Instructions:
======================

Step 1: Create virtual environment
python -m venv bilstm_env
source bilstm_env/bin/activate  # Linux/Mac
bilstm_env\Scripts\activate     # Windows

Step 2: Install TensorFlow carefully (IMPORTANT)
pip install tensorflow==2.12.0 --no-deps

Step 3: Install remaining dependencies
pip install numpy==1.23.5 pandas==1.5.3 scikit-learn==1.2.2 matplotlib==3.7.1

Step 4: Test installation
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

Troubleshooting:
- If you get AVX/JAX errors: Use --no-deps flag with tensorflow
- Mac users: Ensure you're using the correct Python architecture
- If tensorflow==2.12.0 not available: Try tensorflow>=2.12.0
