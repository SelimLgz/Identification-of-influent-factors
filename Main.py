import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the data
df = pd.read_csv("Le Coin Stat Projet/Projet 3/Data.csv", encoding="latin1")

# 1. Quick overview
print(df.info())
print(df.describe())
print(df.head())

# 2. Top products, regions, categories, customer segments
top_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
top_regions = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
top_categories = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
top_segments = df.groupby('Segment')['Sales'].sum().sort_values(ascending=False)

print("Top products:\n", top_products)
print("Sales by region:\n", top_regions)
print("Sales by category:\n", top_categories)
print("Sales by segment:\n", top_segments)

# 3. Visualizations
plt.figure(figsize=(10,5))
sns.barplot(x=top_products.values, y=top_products.index, palette="viridis")
plt.title("Top 10 Products by Sales")
plt.xlabel("Sales")
plt.show()

plt.figure(figsize=(8,4))
sns.barplot(x=top_regions.index, y=top_regions.values, palette="mako")
plt.title("Sales by Region")
plt.ylabel("Sales")
plt.show()

plt.figure(figsize=(8,4))
sns.barplot(x=top_categories.index, y=top_categories.values, palette="Set2")
plt.title("Sales by Category")
plt.ylabel("Sales")
plt.show()

plt.figure(figsize=(8,4))
sns.barplot(x=top_segments.index, y=top_segments.values, palette="Set1")
plt.title("Sales by Customer Segment")
plt.ylabel("Sales")
plt.show()

# 4. Profit analysis by category and region
profit_by_cat = df.groupby('Category')['Profit'].sum().sort_values()
profit_by_region = df.groupby('Region')['Profit'].sum().sort_values()

print("Profit by category:\n", profit_by_cat)
print("Profit by region:\n", profit_by_region)

# 5. Categories or regions to avoid (negative profit)
print("Categories to avoid (negative profit):", profit_by_cat[profit_by_cat < 0])
print("Regions to avoid (negative profit):", profit_by_region[profit_by_region < 0])



# --- Bivariate Analysis: Correlations, Crosstabs, Mean Comparisons ---

# 1. Correlation matrix for numerical variables
num_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
corr_matrix = df[num_cols].corr()
print("Correlation matrix:\n", corr_matrix)

# Heatmap: Correlation matrix
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()

# 2. Crosstab: Category vs. Region (sum of Sales)
crosstab_cat_region = pd.crosstab(df['Category'], df['Region'], values=df['Sales'], aggfunc='sum', normalize='columns')
print("Crosstab (Category vs Region, normalized sales):\n", crosstab_cat_region)

# 3. Mean comparison: Average profit by Segment
mean_profit_by_segment = df.groupby('Segment')['Profit'].mean()
print("Average profit by segment:\n", mean_profit_by_segment)

# --- Advanced Visualizations ---

# Scatter plot: Sales vs Profit, colored by Category
# Interactive Scatter plot using Plotly
fig = px.scatter(
    df,
    x='Sales',
    y='Profit',
    color='Category',
    hover_data=['Product ID', 'Product Name'],  # Infos affichées au survol
    title="Sales vs Profit by Category"
)
fig.show()
# Boxplot: Profit by Region
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Region', y='Profit', palette="Set3")
plt.title("Profit Distribution by Region")
plt.show()



