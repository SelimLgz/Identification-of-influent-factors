import pandas as pd

# Show the sheet names in the Excel file
df = pd.read_csv("Le Coin Stat Projet/Projet 2/HRDataset_v14.csv")
# Display the first few rows of the dataframe
print(df[['Sex','GenderID']].head())

# Missing values in each column
print(df.isnull().sum())

import matplotlib.pyplot as plt

# Percentage of men/women in the dataset
nb_hommes = df['Sex'][df['Sex'] == 'M '].count()
nb_femmes = df['Sex'][df['Sex'] == 'F'].count()

labels = ['Men', 'Women']
sizes = [nb_hommes, nb_femmes]
colors = ["#130c8f", "#dd1791"]

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Répartition Men / Women')
plt.axis('equal') 
plt.show()

# Average salary by gender
avg_salary = df.groupby('Sex')['Salary'].mean()
avg_salary = avg_salary.reindex(['M ', 'F'])  # Ordre: Hommes, Femmes

print(f"Salaires moyens par genre:\n{avg_salary}")

# Show the average salary by gender in a bar chart
avg_salary.index = ['Hommes', 'Femmes']
avg_salary.plot(kind='bar', color=colors)
plt.title('Average salary by gender')
plt.xlabel('Gender')
plt.ylabel('Average Salary')
plt.xticks(rotation=0)
plt.show()

import numpy as np

# Liste des colonnes à ignorer (identifiants, noms, dates, etc.)
ignore_cols = [
    col for col in df.columns if 'ID' in col
] + [
    'Employee_Name', 'DOB', 'DateofHire', 'DateofTermination', 
    'LastPerformanceReview_Date', 'ManagerName', 'TermReason'
]

# Sélection des colonnes à tracer
cols_to_plot = [col for col in df.columns if col not in ignore_cols]

n_cols = 4  # Nombre de colonnes de sous-graphiques
n_rows = int(np.ceil(len(cols_to_plot) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4))
axes = axes.flatten()

for idx, col in enumerate(cols_to_plot):
    ax = axes[idx]
    if np.issubdtype(df[col].dtype, np.number):
        # Histogramme pour les colonnes numériques
        ax.hist(df[col].dropna(), bins=20, color='#130c8f', alpha=0.7)
        ax.set_title(f'Histogramme de {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Effectif')
    else:
        # Barplot pour les colonnes catégorielles
        counts = df[col].value_counts().sort_index()
        if len(counts) > 30:
            ax.axis('off')
            continue
        counts.plot(kind='bar', color='#dd1791', alpha=0.7, ax=ax)
        ax.set_title(f'Barplot de {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Effectif')

# Désactive les axes inutilisés
for i in range(len(cols_to_plot), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()

def detect_outliers_iqr(series):
    """Détection des outliers par la méthode des 1.5*IQR."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series[(series < lower) | (series > upper)].index

def detect_outliers_zscore(series, threshold=3):
    """Détection des outliers par la méthode du Z-score."""
    mean = series.mean()
    std = series.std()
    z_scores = (series - mean) / std
    return series[(z_scores.abs() > threshold)].index

# Colonnes numériques à analyser (hors ID)
num_cols = [col for col in df.select_dtypes(include=np.number).columns if 'ID' not in col]

outliers_confirmed = {}

for col in num_cols:
    iqr_idx = set(detect_outliers_iqr(df[col].dropna()))
    zscore_idx = set(detect_outliers_zscore(df[col].dropna()))
    # Intersection : outliers confirmés par les deux méthodes
    confirmed_idx = iqr_idx & zscore_idx
    if confirmed_idx:
        outliers_confirmed[col] = confirmed_idx

# Affichage des résultats
for col, idxs in outliers_confirmed.items():
    print(f"Outliers confirmés pour {col} (IQR & Z-score):")
    print(df.loc[list(idxs), [col]])