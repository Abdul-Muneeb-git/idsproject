import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load and inspect the dataset"""
    try:
        df = pd.read_csv(file_path)
        logger.info("Data loaded successfully. Shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        raise

def clean_data(df):
    """Clean and preprocess the data"""
    try:
        # Convert percentage strings to numeric values
        percent_cols = [col for col in df.columns if 'percent' in col.lower() or '%' in col]
        for col in percent_cols:
            if df[col].dtype == object:
                df[col] = df[col].str.rstrip('%').astype('float') / 100.0
        
        # Clean currency columns with proper regex escaping
        currency_cols = ['stock_price', 'price']
        for col in currency_cols:
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].replace(r'[\$,]', '', regex=True).astype(float)
        
        # Clean market capitalization
        if 'market_capitalization' in df.columns:
            if df['market_capitalization'].dtype == object:
                df['market_capitalization'] = df['market_capitalization'].replace(r'[\$,]', '', regex=True).astype(float)
        
        # Ensure momentum_score is numeric
        if 'momentum_score' in df.columns:
            if df['momentum_score'].dtype == object:
                df['momentum_score'] = pd.to_numeric(df['momentum_score'], errors='coerce')
        
        logger.info("Data cleaning completed")
        return df
    except Exception as e:
        logger.error("Error during data cleaning: %s", e)
        raise

def handle_missing_values(df, strategy='median'):
    """Handle missing values with specified strategy"""
    try:
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Identify columns with all NaN values
        all_nan_cols = [col for col in numeric_cols if df[col].isnull().all()]
        if all_nan_cols:
            logger.warning("Columns with all NaN values: %s. These will be filled with 0.", all_nan_cols)
            df[all_nan_cols] = df[all_nan_cols].fillna(0)
            numeric_cols = [col for col in numeric_cols if col not in all_nan_cols]
        
        # Impute remaining numeric columns
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy=strategy)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # For categorical columns, we'll fill with 'Unknown'
        if len(categorical_cols) > 0:
            df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        
        logger.info("Missing values handled using %s strategy", strategy)
        return df
    except Exception as e:
        logger.error("Error handling missing values: %s", e)
        raise

def feature_engineering(df):
    """Create new features from existing ones"""
    try:
        # Create a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Only work with numeric columns for feature engineering
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Calculate price to momentum score ratio if columns exist
        if 'momentum_score' in numeric_cols and 'stock_price' in numeric_cols:
            df['price_to_momentum'] = df['stock_price'] / (df['momentum_score'].replace(0, 0.001) + 0.001)
        
        # Calculate value to growth ratio (if both scores exist)
        if 'value_score' in numeric_cols and 'momentum_score' in numeric_cols:
            df['value_growth_ratio'] = df['value_score'] / (df['momentum_score'].replace(0, 0.001) + 0.001)
        
        # Market cap categories (only if market_capitalization is numeric)
        if 'market_capitalization' in numeric_cols:
            df['market_cap_category'] = pd.cut(
                df['market_capitalization'],
                bins=[0, 2e9, 1e10, 1e11, float('inf')],
                labels=['Small', 'Mid', 'Large', 'Mega']
            )
        
        logger.info("Feature engineering completed")
        return df
    except Exception as e:
        logger.error("Error during feature engineering: %s", e)
        raise

def select_features(df, target_col='momentum_score', k=20):
    """Select top k features based on correlation with target"""
    try:
        # Ensure target exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Check if target column has any non-NaN values
        if df[target_col].isnull().all():
            logger.warning("Target column '%s' is completely empty. Using all features.", target_col)
            return df
        
        # Convert target to numeric if needed
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            if df[target_col].isnull().all():
                logger.warning("Target column '%s' could not be converted to numeric. Using all features.", target_col)
                return df
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Select numeric features only
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        X_numeric = X[numeric_cols]
        
        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(k, len(numeric_cols)))
        selector.fit(X_numeric, y)
        
        # Get selected features
        selected_features = X_numeric.columns[selector.get_support()]
        logger.info("Selected top %d features: %s", k, list(selected_features))
        
        # Include non-numeric columns that might be important (like categories)
        non_numeric_cols = X.select_dtypes(exclude=['float64', 'int64']).columns
        return df[list(selected_features) + list(non_numeric_cols) + [target_col]]
    except Exception as e:
        logger.error("Error during feature selection: %s", e)
        raise

def scale_features(df, target_col=None, scaler_type='standard'):
    """Scale features using specified scaler"""
    try:
        # Create a copy to avoid modifying the original dataframe
        df = df.copy()
        
        if target_col and target_col in df.columns:
            features = df.drop(columns=[target_col])
            target = df[[target_col]]
        else:
            features = df
            target = None
            
        # Select numeric features only for scaling
        numeric_cols = features.select_dtypes(include=['float64', 'int64']).columns
        non_numeric_cols = features.select_dtypes(exclude=['float64', 'int64']).columns
        
        # Initialize scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler type. Choose 'standard' or 'minmax'")
        
        # Scale only numeric features
        if len(numeric_cols) > 0:
            features_numeric = features[numeric_cols]
            scaled_features = scaler.fit_transform(features_numeric)
            features[numeric_cols] = scaled_features
        
        # Combine back with non-numeric columns
        if len(non_numeric_cols) > 0:
            features = pd.concat([features[numeric_cols], features[non_numeric_cols]], axis=1)
        
        if target is not None:
            df = pd.concat([features, target], axis=1)
        else:
            df = features
            
        logger.info("Features scaled using %s scaler", scaler_type)
        return df, scaler
    except Exception as e:
        logger.error("Error during feature scaling: %s", e)
        raise

def preprocess_pipeline(file_path, target_col='momentum_score'):
    """Complete preprocessing pipeline"""
    try:
        logger.info("Starting preprocessing pipeline")
        
        # 1. Load data
        df = load_data(file_path)
        
        # 2. Clean data
        df = clean_data(df)
        
        # 3. Handle missing values
        df = handle_missing_values(df)
        
        # 4. Feature engineering
        df = feature_engineering(df)
        
        # 5. Feature selection
        df = select_features(df, target_col)
        
        # 6. Scale features
        df, scaler = scale_features(df, target_col)
        
        logger.info("Preprocessing pipeline completed successfully")
        return df, scaler
    except Exception as e:
        logger.error("Preprocessing pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    # Example usage
    input_path = "merged_stock_data.csv"
    
    try:
        # Run preprocessing pipeline
        processed_data, scaler = preprocess_pipeline(input_path, target_col='momentum_score')
        
        # Save processed data
        processed_data.to_csv("processed_stock_data.csv", index=False)
        logger.info("Processed data saved to processed_stock_data.csv")
        
        # Show sample of processed data
        print("\nSample of processed data:")
        print(processed_data.head())
        
    except Exception as e:
        logger.error("Error in main execution: %s", e)