# 🧬 ML Cancer Detection Pipeline

A complete end-to-end machine learning pipeline for breast cancer classification using scikit-learn, deployed as a REST API with Flask.

## Overview

This project demonstrates a production-ready ML workflow — from raw data exploration and preprocessing to model training, evaluation, and deployment. It classifies tumors as **benign** or **malignant** using 30 clinical features with over 96% accuracy.

## Tech Stack

- **Python** — core language
- **scikit-learn** — model training and evaluation
- **Flask** — REST API deployment
- **pandas / numpy** — data manipulation
- **pickle** — model serialization
- **Docker** — containerization

## Project Structure

```
ml-cancer-detection-pipeline/
│
├── pipeline.py          # Full ML pipeline (EDA → train → evaluate → save)
├── app.py               # Flask REST API serving predictions
├── requirements.txt     # Dependencies
├── Dockerfile           # Container setup
└── README.md
```

## ML Pipeline

The pipeline follows a complete SDLC-aligned workflow:

```
Raw Data → EDA → Preprocessing → Model Training → Evaluation → Save → Deploy
```

### 1. Exploratory Data Analysis
- Dataset shape, class distribution, missing value checks
- Feature correlation analysis to identify most predictive variables

### 2. Preprocessing
- Train/test split (80/20) with stratification to preserve class balance
- StandardScaler normalization — fit on train only to prevent data leakage

### 3. Model Training
Two models trained and compared automatically:
- **Logistic Regression** — fast, interpretable baseline
- **Random Forest** — ensemble method for complex patterns

Best model selected automatically based on test accuracy.

### 4. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix with breakdown of error types
- Feature importance ranking (Random Forest)
- 5-fold cross-validation for reliable performance estimate

### 5. Model Persistence
Best model, scaler, and feature names saved via pickle for API use.

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~95% |
| Random Forest | ~96–97% |

> Note: Random Forest typically wins and is auto-selected as the deployed model.

## Setup & Installation

### Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ml-cancer-detection-pipeline.git
cd ml-cancer-detection-pipeline
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python pipeline.py
```

This generates `best_model.pkl`, `scaler.pkl`, and `features.pkl`.

### Run the API
```bash
python app.py
```

Server starts at `http://localhost:5000`

## API Usage

### `GET /`
Returns API info and list of required features.

### `POST /predict`
Accepts 30 clinical features and returns prediction with confidence.

**Request:**
```json
{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
               0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
               0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
               25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
               0.2654, 0.4601, 0.1189]
}
```

**Response:**
```json
{
  "prediction": "Malignant",
  "confidence": "97.3%",
  "probabilities": {
    "malignant": "97.3%",
    "benign": "2.7%"
  }
}
```

## Docker

```bash
# Build image
docker build -t ml-cancer-api .

# Run container
docker run -p 5000:5000 ml-cancer-api
```

## Key Concepts Demonstrated

- **Data leakage prevention** — scaler fitted on training data only
- **Stratified splitting** — preserves class balance across train/test
- **Model comparison** — automatic selection of best performer
- **Soft delete pattern** — extended to API design principles
- **Cross-validation** — reliable accuracy estimate beyond single split
- **Feature importance** — identifying most predictive clinical variables
- **Proper evaluation** — precision, recall, F1 alongside accuracy

## Why These Metrics Matter

Accuracy alone is misleading for medical data. A model that always predicts "benign" achieves ~63% accuracy on this dataset while missing every cancer case. This project tracks **recall** (catching actual malignant cases) as the primary metric — because a false negative in cancer detection is far more costly than a false positive.

## Dataset

UCI Breast Cancer Wisconsin Dataset — included via `sklearn.datasets.load_breast_cancer()`. 569 samples, 30 features, 2 classes (357 benign, 212 malignant).

## Author

**Fakhir Shahzad**  
BS Computer Science — FAST-NUCES Lahore (2024–2028)  
[l240512@lhr.nu.edu.pk](mailto:l240512@lhr.nu.edu.pk)
