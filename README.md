


# CRTDiff: Conditional Diffusion for PPV Prediction

This repository provides the implementation of **CRTDiff**, a conditional diffusion-based data generation method designed for enhancing peak particle velocity (PPV) prediction in open-pit blasting scenarios.

## 📌 Features

- Conditional diffusion sampling based on DDPM
- Handles mixed discrete-continuous variables
- Residual network with sinusoidal time embedding
- Data augmentation for supervised ML models (e.g., ANN, SVR, XGBoost)
- SHAP-ready feature interpretability

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `train_crtdiff.py` | Main training loop with loss tracking |
| `model.py` | Model architecture (ResNet + Time Embedding) |
| `data_utils.py` | Data loading, encoding, normalization |
| `generate_samples.py` | Sampling from trained model |
| `config.py` | Configuration of hyperparameters and paths |


```

## 📄 Dataset

The default dataset is `origina data.xlsx`. You may replace it with your own structured blasting dataset, formatted as columns `X1 ~ X11 + PPV`.

## 📜 License

📚 Citation

If you find this work helpful in your research, please consider citing the following paper:

Li, H., Xie, B., Li, X. et al.
CRTDiff: A Conditional Residual Temporal Diffusion Model for Data Augmentation to Enhance Machine Learning Prediction of PPV in Open-Pit Mining.
Rock Mechanics and Rock Engineering (2025).
https://doi.org/10.1007/s00603-025-05002-9

MIT License © 2024
