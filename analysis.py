# Stock Data EDA - Complete Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# 1. Load and Prepare Data
df = pd.read_csv('merged_stock_data.csv')

# Convert percentage strings to numeric values
percent_cols = [col for col in df.columns if 'percent' in col.lower() or 'return' in col.lower()]
for col in percent_cols:
    if df[col].dtype == 'object':
        df[col] = df[col].str.rstrip('%').astype('float') / 100.0

# Clean price columns (remove $ and commas)
price_cols = ['stock_price', 'price']
for col in price_cols:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].replace(r'[\$,]', '', regex=True).astype(float)

# 2. Basic Statistics
print("="*50)
print("BASIC STATISTICS")
print("="*50)
print("\nNumerical Columns Summary:")
print(df.describe())

print("\nCategorical Columns Summary:")
print(df.describe(include=['object']))

# 3. Missing Value Analysis
print("\n" + "="*50)
print("MISSING VALUE ANALYSIS")
print("="*50)
missing_data = df.isnull().sum().sort_values(ascending=False)
missing_percent = (df.isnull().sum() / len(df)).sort_values(ascending=False)
print(pd.concat([missing_data, missing_percent], axis=1, 
               keys=['Total Missing', 'Percent Missing']).head(20))

# Visualization Settings
print("Available styles:", plt.style.available)  # Optional: See all choices
plt.style.use('seaborn-v0_8')  # Or 'ggplot', 'tableau-colorblind10', etc.
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_palette("husl")

# 5. Key Visualizations
# a. Score Distributions
print("\n" + "="*50)
print("VISUALIZATIONS")
print("="*50)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['momentum_score'].dropna(), kde=True, bins=20)
plt.title('Momentum Score Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df['value_score'].dropna(), kde=True, bins=20)
plt.title('Value Score Distribution')
plt.tight_layout()
plt.show()

# CORRELATION ANALYSIS (FIXED VERSION)
# =============================================

# 1. Define columns for correlation analysis
corr_cols = [
    'momentum_score', 
    'value_score', 
    'one_year_return_percentile',
    'six_month_return_percentile', 
    'three_month_return_percentile',
    'one_month_return_percentile',
    'price-to-earnings_ratio',
    'price-to-book_ratio',
    'price-to-sales_ratio',
    'market_capitalization'
]

# 2. Clean and convert numeric columns
for col in corr_cols:
    if col in df.columns:
        # Remove % and $ signs, convert to numeric
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('[%,$]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Calculate correlation matrix
corr_matrix = df[corr_cols].corr()

# 4. Plot enhanced heatmap
plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show only lower triangle
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title('Stock Metrics Correlation Matrix\n(Values show Pearson correlation coefficients)', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# c. Return Trends Comparison
time_periods = ['one_year', 'six_month', 'three_month', 'one_month']
returns = [f'{period}_price_return' for period in time_periods]

plt.figure(figsize=(10, 6))
df[returns].mean().plot(kind='bar')
plt.title('Average Returns Across Time Periods')
plt.ylabel('Average Return')
plt.xticks(rotation=45)
plt.show()

# d. Price vs Market Cap Analysis
plt.figure(figsize=(12, 6))
sns.scatterplot(x='stock_price', y='market_capitalization', data=df, hue='momentum_score')
plt.title('Stock Price vs Market Capitalization')
plt.xscale('log')
plt.yscale('log')
plt.show()

# 6. Save Cleaned Data (Optional)
df.to_csv('cleaned_stock_data.csv', index=False)
print("\nEDA complete! Cleaned data saved to 'cleaned_stock_data.csv'")