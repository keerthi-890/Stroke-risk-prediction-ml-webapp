# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Select categorical columns relevant to the base paper
categorical_columns = [
    'gender',
    'work_type',
    'hypertension',
    'Residence_type',
    'heart_disease',
    'smoking_status',
    'ever_married',
    'stroke'
]

# Define custom titles (prettier names)
titles = {
    'gender': 'GENDER',
    'work_type': 'WORK TYPE',
    'hypertension': 'HYPERTENSION',
    'Residence_type': 'RESIDENCE TYPE',
    'heart_disease': 'HEART DISEASE',
    'smoking_status': 'SMOKING STATUS',
    'ever_married': 'MARITAL STATUS',
    'stroke': 'STROKE'
}

# Define color palette for consistent visuals
color_sets = [
    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'],
    ['#66b3ff', '#99ff99'],
    ['#ffb366', '#999999'],
    ['#87CEFA','#4169E1'],
    ['#ff7f0e','#2ca02c','#8c564b','#7f7f7f'],
    ['#FFD580','#FFB347'],
    ['#FFA500','#FF8C00']
]

# Create subplots grid
fig, axes = plt.subplots(4, 2, figsize=(10, 12))
axes = axes.flatten()

# Loop through each categorical column and create pie chart
for i, col in enumerate(categorical_columns):
    data = df[col].value_counts()
    colors = color_sets[i % len(color_sets)]
    axes[i].pie(data, labels=data.index, autopct='%1.0f%%', startangle=90, colors=colors[:len(data)])
    axes[i].set_title(titles[col], fontsize=12, fontweight='bold')

# Adjust spacing between plots
plt.tight_layout()
plt.show()