# Assignment 1: Insurance Charges Prediction

## Overview
This project implements a machine learning pipeline to predict medical insurance charges based on beneficiary attributes (Regression) and classify them into cost tiers (Classification). It compares 11 different algorithms, including linear models, KNN, and advanced boosting techniques.

**Dataset**: [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)

## Project Structure
- `assignment1.py`: Main script containing the training and evaluation logic.
- `REPORT.md`: Detailed analysis and performance table.
- `insurance.csv`: Dataset file.

## How to Run

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Execution
Run the assignment script using `uv`:
```bash
uv run assignment1.py
```

Or using standard python (after installing dependencies):
```bash
pip install -r requirements.txt
python assignment1.py
```

## Results Summary
The project evaluated 11 models using 10-fold cross-validation.

- **Best Model**: `GradientBoostingRegressor/Classifier`
- **Regression Performance**: RMSE: **4517.51**, R²: **0.8539**
- **Classification Performance**: Accuracy: **89.61%**

For the detailed comparison table and analysis, please refer to [REPORT.md](REPORT.md).
