# Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load your data (replace with your file path)
df = pd.read_csv('processed_stock_data.csv')

# =====================================
# STEP 1: Clean & Prepare Data
# =====================================

# Fix 'value_score' (convert % to numbers, handle 'Unknown')
df['value_score'] = df['value_score'].replace('Unknown', np.nan)
df['value_score'] = df['value_score'].str.replace('%', '').astype(float)

# Drop rows with missing target (if needed)
df = df.dropna(subset=['value_score'])

# Define features (X) and target (y)
features = [
    'one_year_return_percentile', 'six_month_return_percentile',
    'three_month_return_percentile', 'one_month_return_percentile',
    'price-to-earnings_ratio', 'pe_percentile', 'price-to-book_ratio',
    'pb_percentile', 'price-to-sales_ratio', 'ps_percentile',
    'ev/ebitda', 'ev/ebitda_percentile', 'ev/gp', 'ev/gp_percentile',
    'momentum_score'
]

X = df[features]  # Features
y = (df['value_score'] > df['value_score'].median()).astype(int)  # Binary target (1=High Value, 0=Low Value)

# =====================================
# STEP 2: Train Random Forest Model
# =====================================

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =====================================
# STEP 3: Evaluate Performance
# =====================================

# Predict on test data
y_pred = model.predict(X_test)

# Print accuracy & classification report
print("=== Model Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================
# STEP 4: Feature Importance (What Matters Most?)
# =====================================

# Get feature importances
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n=== Top 5 Most Important Features ===")
print(feature_importance.head(5))

# =====================================
# STEP 5: Save Model for Future Use
# =====================================
import joblib
joblib.dump(model, 'stock_value_predictor.pkl')
print("\nModel saved as 'stock_value_predictor.pkl'")