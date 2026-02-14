import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Scatter plot of BMI before imputation
fig, ax = plt.subplots()
non_null_bmi = df['bmi'].notnull()
ax.scatter(df[non_null_bmi].index, df.loc[non_null_bmi, 'bmi'], label='Non-null BMI', s=10)
ax.scatter(df[~non_null_bmi].index, [0]*sum(~non_null_bmi), label='Null BMI', color='red', s=10)
ax.set_xlabel('Index')
ax.set_ylabel('BMI')
ax.legend()
plt.title('Scatter plot of BMI values (before imputation)')
plt.show()

# Impute missing BMI values with mean
imputer = SimpleImputer(strategy='mean')
df['bmi_imputed'] = imputer.fit_transform(df[['bmi']])

# Scatter plot of BMI after imputation
fig, ax = plt.subplots()
ax.scatter(df.index, df['bmi_imputed'], label='Imputed BMI', color='green', s=10)
ax.scatter(df[df['bmi'].notnull()].index, df.loc[df['bmi'].notnull(), 'bmi'], label='Original BMI', alpha=0.5, s=10)
ax.set_xlabel('Index')
ax.set_ylabel('BMI')
ax.legend()
plt.title('Scatter plot of BMI values (after imputation)')
plt.show()

# Drop ID column
df = df.drop(columns=['id'])

# One-hot encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Compute correlation matrix
corr = df_encoded.corr()

# Plot correlation matrix as heatmap
plt.figure(figsize=(10,10))
sns.heatmap(corr, cmap='YlGnBu')
plt.title('Correlation Matrix (with one-hot encoded categorical features)')
plt.show()
