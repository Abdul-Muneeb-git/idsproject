import pandas as pd
import os
from tkinter import Tk, filedialog
from pathlib import Path

def merge_stock_data():
    print("STOCK DATA MERGER")
    print("This script will combine momentum, value, and trading datasets")
    
    # Prepare file paths
    base_dir = Path.cwd()
    momentum_path = base_dir / 'momentum_strategy_1.csv'
    value_path = base_dir / 'value_strategy_1.csv'
    trades_path = base_dir / 'recommended_trades_1.csv'
    
    # Check for files automatically
    if all([f.exists() for f in [momentum_path, value_path, trades_path]]):
        print("\nFound all CSV files in current directory")
        momentum_df = pd.read_csv(momentum_path)
        value_df = pd.read_csv(value_path)
        trades_df = pd.read_csv(trades_path)
    else:
        print("\nSome files missing in current directory. Please select them manually.")
        root = Tk()
        root.withdraw()
        
        print("\nSelect momentum_strategy_1.csv")
        momentum_path = Path(filedialog.askopenfilename(title="Select momentum_strategy_1.csv"))
        
        print("\nSelect value_strategy_1.csv")
        value_path = Path(filedialog.askopenfilename(title="Select value_strategy_1.csv"))
        
        print("\nSelect recommended_trades_1.csv")
        trades_path = Path(filedialog.askopenfilename(title="Select recommended_trades_1.csv"))
        
        momentum_df = pd.read_csv(momentum_path)
        value_df = pd.read_csv(value_path)
        trades_df = pd.read_csv(trades_path)

    # Merge datasets with intelligent column handling
    print("\nMerging datasets...")
    combined = pd.merge(
        momentum_df, 
        value_df, 
        on='Ticker', 
        how='outer',
        suffixes=('_momentum', '_value')
    )
    
    final_df = pd.merge(
        combined,
        trades_df,
        on='Ticker',
        how='outer'
    )
    
    # Clean merged data
    print("Cleaning merged data...")
    final_df.columns = [col.strip().replace(' ', '_').lower() for col in final_df.columns]
    
    # Handle duplicate columns - more robust version
    if 'price_momentum' in final_df.columns and 'price_value' in final_df.columns:
        final_df['price'] = final_df['price_momentum'].fillna(final_df['price_value'])
        final_df.drop(['price_momentum', 'price_value'], axis=1, inplace=True)
    
    # More robust shares_to_buy handling
    shares_cols = [col for col in final_df.columns if 'shares_to_buy' in col.lower() or 'number_of_shares_to_buy' in col.lower()]
    
    if len(shares_cols) > 0:
        # Create new column by coalescing all shares columns
        final_df['shares_to_buy'] = final_df[shares_cols[0]]
        for col in shares_cols[1:]:
            final_df['shares_to_buy'] = final_df['shares_to_buy'].fillna(final_df[col])
        
        # Drop the original columns
        final_df.drop(shares_cols, axis=1, inplace=True)
    
    # Save merged file
    output_path = base_dir / 'merged_stock_data.csv'
    final_df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print(f"MERGED FILE SUCCESSFULLY CREATED AT:\n{output_path.absolute()}")
    print(f"\nMerged dataset contains {len(final_df)} stocks with {len(final_df.columns)} columns")
    print("\nFirst 3 merged records:")
    return final_df.head(3)

# Run the merger
if __name__ == "__main__":
    try:
        result = merge_stock_data()
        print(result)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nPlease ensure:")
        print("1. All input CSV files exist")
        print("2. The files contain a 'Ticker' column")
        print("3. You have write permissions in the current directory")