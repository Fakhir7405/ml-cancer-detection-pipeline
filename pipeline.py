import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                              confusion_matrix)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

# We'll use sklearn's built-in breast cancer dataset
# Real internship: you'd load CSV with pd.read_csv('data.csv')
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Convert to DataFrame (table format)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns[:5])}... and more")
print(f"Target values: {df['target'].value_counts().to_dict()}")
print(f"0 = malignant, 1 = benign")
print(df.head())

print("\n" + "=" * 60)
print("STEP 2: EXPLORING DATA (EDA)")
print("=" * 60)

# Basic stats
print("\nBasic Statistics:")
print(df.describe().round(2))

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Class balance
print(f"\nClass distribution:")
print(df['target'].value_counts())
print(f"Benign: {(df['target']==1).sum()} | Malignant: {(df['target']==0).sum()}")

# Correlation with target
print(f"\nTop 5 features correlated with target:")
corr = df.corr()['target'].abs().sort_values(ascending=False)
print(corr[1:6])


print("\n" + "=" * 60)
print("STEP 3: PREPROCESSING")
print("=" * 60)

# Separate features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split into train and test sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% test
    random_state=42,    # same split every time
    stratify=y          # keep class balance in both splits
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Scale features
# Many algorithms need features on same scale
# Without this: a feature with range 0-1000 dominates one with 0-1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_test_scaled = scaler.transform(X_test)          # ONLY transform on test

print("✅ Data scaled successfully")


print("\n" + "=" * 60)
print("STEP 4: TRAINING MODELS")
print("=" * 60)

# Train Model 1: Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.1f}%)")

# Train Model 2: Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.1f}%)")

from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Pick best model
if rf_accuracy > lr_accuracy:
    best_model = rf_model
    best_pred = rf_pred
    best_name = "Random Forest"
else:
    best_model = lr_model
    best_pred = lr_pred
    best_name = "Logistic Regression"

print(f"\n🏆 Best Model: {best_name} with {max(rf_accuracy, lr_accuracy)*100:.1f}% accuracy")

print("\n" + "=" * 60)
print("STEP 5: EVALUATION")
print("=" * 60)

print(f"\nDetailed Report for {best_name}:")
print(classification_report(y_test, best_pred, 
                             target_names=['Malignant', 'Benign']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(cm)
print(f"""
Breakdown:
True Negatives  (correctly said malignant): {cm[0][0]}
False Positives (said benign, was malignant): {cm[0][1]}  ← DANGEROUS
False Negatives (said malignant, was benign): {cm[1][0]}
True Positives  (correctly said benign): {cm[1][1]}
""")

print("\n" + "=" * 60)
print("STEP 6: SAVING MODEL")
print("=" * 60)

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names for later use
feature_names = list(X.columns)
with open('features.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print(f"✅ {best_name} saved as best_model.pkl")
print(f"✅ Scaler saved as scaler.pkl")
print(f"✅ Feature names saved as features.pkl")





# IMPORTANT ML CONCEPTS:
"""
1) Overfitting vs Underfitting (bias-variance tradeoff)
2) Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
3) Cross-validation (KFold, StratifiedKFold)
4) Feature importance 

"""