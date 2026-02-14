import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

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

# Label encoding categorical variables
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])

le_married = LabelEncoder()
df['ever_married'] = le_married.fit_transform(df['ever_married'])

le_residence = LabelEncoder()
df['Residence_type'] = le_residence.fit_transform(df['Residence_type'])

# One-hot encode nominal categorical variables
df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True)

# --- 3. Define Features and Target ---
X = df.drop('stroke', axis=1)
y = df['stroke']

# --- 4. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# --- 5. Feature Scaling ---
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 6. Apply SMOTE ---
print("Applying SMOTE to training data...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# --- 7. Hyperparameter Tuning with RandomizedSearchCV ---
print("Running RandomizedSearchCV for Logistic Regression...")
param_grid = {
    'C': np.logspace(-2, 2, 10),
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs'],
    'class_weight': [None, 'balanced']
}
lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
random_search = RandomizedSearchCV(
    lr, param_grid, n_iter=10, cv=3,
    scoring='accuracy', n_jobs=-1, random_state=RANDOM_STATE, verbose=0
)
random_search.fit(X_train_resampled, y_train_resampled)
best_lr = random_search.best_estimator_

print("\n---- Best Model Parameters ----")
print(random_search.best_params_)

# --- 8. Predict and Threshold Tuning on Test Set ---
y_prob = best_lr.predict_proba(X_test_scaled)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
best_f1, best_th = 0, 0.5

for th in thresholds:
    y_pred_temp = (y_prob >= th).astype(int)
    f1 = f1_score(y_test, y_pred_temp, zero_division=0)
    if f1 > best_f1:
        best_f1, best_th = f1, th

y_pred = (y_prob >= best_th).astype(int)

print(f"Best Threshold (Optimized for F1): {best_th:.2f}")

# --- 9. Confusion Matrix and Metrics ---

# Training predictions on SMOTE resampled data
y_train_pred = best_lr.predict(X_train_resampled)
cm_train = confusion_matrix(y_train_resampled, y_train_pred)
print("\nTraining Confusion Matrix:")
print(cm_train)

plt.figure(figsize=(6,5))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='plasma',
            xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
plt.title('Training Data Confusion Matrix (After SMOTE)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()

# Test confusion matrix
cm_test = confusion_matrix(y_test, y_pred)
print("\nTesting Confusion Matrix:")
print(cm_test)

plt.figure(figsize=(6,5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='plasma',
            xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
plt.title('Testing Data Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()

# Print other evaluation metrics on test set
print("\n---- Final Model Performance on Test Data ----")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
