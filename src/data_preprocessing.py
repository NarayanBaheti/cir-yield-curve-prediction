import numpy as np
import pandas as pd

def preprocess_yields(train_df, test_df):
    """
    Trims headers, parses dates, sorts chronologically, interpolates missing values,
    and applies Z-score Winsorization at 3-sigma.
    """
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()
    
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    train_df = train_df.sort_values('Date').reset_index(drop=True)
    test_df = test_df.sort_values('Date').reset_index(drop=True)
    
    yield_cols = [col for col in train_df.columns if col != 'Date']
    train_df[yield_cols] = train_df[yield_cols].interpolate(method='linear').ffill().bfill()
    
    test_yield_cols = [col for col in test_df.columns if col != 'Date']
    test_df[test_yield_cols] = test_df[test_yield_cols].interpolate(method='linear').ffill().bfill()
    
    train_clean = train_df.copy()
    test_clean = test_df.copy()
    
    for col in yield_cols:
        mean_val = train_df[col].mean()
        std_val = train_df[col].std()
        upper_limit = mean_val + 3 * std_val
        lower_limit = mean_val - 3 * std_val
        train_clean[col] = np.clip(train_df[col], lower_limit, upper_limit)
        if col in test_clean.columns:
            test_clean[col] = np.clip(test_df[col], lower_limit, upper_limit)
            
    return train_clean, test_clean
