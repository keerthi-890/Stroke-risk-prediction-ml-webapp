import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import shap
import matplotlib.pyplot as plt


# --- 1. Configuration and Data Loading ---
DATA_FILE = 'healthcare-dataset-stroke-data.csv'
RANDOM_STATE = 42

print("---- Starting Stroke Prediction Pipeline ----")
print(f"Loading data from: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

# --- 2. Data Cleaning and Preprocessing ---

if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

df.dropna(subset=['stroke'], inplace=True)
df = df[df['gender'] != 'Other']

df['bmi'] = df['bmi'].replace('N/A', np.nan).astype(float)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])

le_married = LabelEncoder()
df['ever_married'] = le_married.fit_transform(df['ever_married'])

le_residence = LabelEncoder()
df['Residence_type'] = le_residence.fit_transform(df['Residence_type'])

df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True)

# --- 4. Define Features and Target ---
X = df.drop('stroke', axis=1)
y = df['stroke']

# --- 5. Train-Test Split and Save CSVs ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Save train/test splits as CSV
train_df = X_train.copy()
train_df['stroke'] = y_train
test_df = X_test.copy()
test_df['stroke'] = y_test
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
print("Train and test splits saved as 'train.csv' and 'test.csv'.")

# --- 6. Scaling (Standardization) ---
print("Scaling features...")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("Applying SMOTE to training data...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# --- 8. Model Training with Hyperparameter Tuning (RandomizedSearchCV) ---
print("Running RandomizedSearchCV for Logistic Regression...")

param_grid = {
    'C': np.logspace(-2, 2, 10),  # 0.01 to 100
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs'],
    'class_weight': [None, 'balanced']
}

lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)

random_search = RandomizedSearchCV(
    lr, param_grid, n_iter=10, cv=3,
    scoring='accuracy', n_jobs=-1, random_state=RANDOM_STATE, verbose=0
)
# Train on the SMOTE-resampled and scaled data
random_search.fit(X_train_resampled, y_train_resampled)

best_lr = random_search.best_estimator_

# --- 9. Evaluation and Threshold Tuning ---

print("\n---- Best Model Parameters and Performance ----")
print("Best parameters:", random_search.best_params_)

# Predict probabilities on the scaled test set
y_prob = best_lr.predict_proba(X_test_scaled)[:, 1]

# Auto threshold tuning (optional but useful for imbalanced data)
thresholds = np.arange(0.1, 0.9, 0.01) # Check a wider range
best_f1, best_th = 0, 0.5 # Optimizing for F1-score is often better for imbalanced classification
best_acc = 0 # Also track accuracy
for th in thresholds:
    y_pred_temp = (y_prob >= th).astype(int)
    f1 = f1_score(y_test, y_pred_temp, zero_division=0)
    acc = accuracy_score(y_test, y_pred_temp)

    if f1 > best_f1:
        best_f1, best_th = f1, th
        best_acc = acc # Update accuracy corresponding to best F1

# Final predictions using the optimal threshold
y_pred = (y_prob >= best_th).astype(int)

print(f"Best Threshold (Optimized for F1): {best_th:.2f}")

# Metrics
print("\n---- Final Model Performance on Test Data ----")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f} (at best F1 threshold)")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print("\nConfusion Matrix (Test Data):")
print(confusion_matrix(y_test, y_pred))

# --- 11. Explainability (SHAP) ---
print("\nGenerating Explainable AI (XAI) insights...")

# Convert scaled test data back to DataFrame (for SHAP)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# --- SHAP EXPLANATION ---
explainer_shap = shap.LinearExplainer(best_lr, X_train_scaled, feature_names=X.columns)
shap_values = explainer_shap.shap_values(X_test_scaled)

# SHAP summary plot (overall feature impact)
plt.figure()
shap.summary_plot(shap_values, X_test_scaled_df, show=False)
plt.title("SHAP Summary - Stroke Prediction Model")
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
plt.close()
print("✅ SHAP summary plot saved as 'shap_summary_plot.png'")


# --- 10. Save Artifacts ---

print("\nSaving model artifacts...")

# Save feature names
with open('features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

# Save the best trained model
with open('stroke_lr_model.pkl', 'wb') as f:
    pickle.dump(best_lr, f)

# Save the fitted scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save encoders
encoders = {
    'gender': le_gender,
    'ever_married': le_married,
    'Residence_type': le_residence
}
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("Model, scaler, encoders, and feature names saved successfully!")
print("\n✅ Finished Stroke Prediction Run")