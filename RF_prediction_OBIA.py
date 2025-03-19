import os
import glob
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.metrics import r2_score

# Define data directory (Modify before running)
data_dir = "path/to/data"

# Load CSV files dynamically
pin_files = glob.glob(os.path.join(data_dir, "2019_text/Pink_finescale_new/txtgeo/*.csv"))
zou_files = glob.glob(os.path.join(data_dir, "2019_text/Zout_finescale_new/txtgeo/*.csv"))

# Initialize results storage
df_h = pd.DataFrame()

# Loop through each dataset
for i, pin_file in enumerate(pin_files):
    # Load data
    df_p = pd.read_csv(pin_file)
    df_z = pd.read_csv(zou_files[i])
    
    # Standardize column names
    df_p.columns = range(df_p.shape[1])
    df_z.columns = range(df_z.shape[1])
    
    # Merge datasets
    combined_df = pd.concat([df_p, df_z], ignore_index=True, sort=False)
    
    # Select relevant features
    combined_df = combined_df.iloc[:, 6:]
    combined_df = combined_df[combined_df.iloc[:, 2] != 0]  # Remove zero values
    
    # Define target variable
    y = np.log10(combined_df.iloc[:, 2])
    X = combined_df.iloc[:, 7:]
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=800, random_state=42, max_depth=20, 
                                  max_features='sqrt', min_samples_split=10, 
                                  min_samples_leaf=2, bootstrap=False)
    
    # Perform feature selection
    selector = sfs(model, k_features=30, forward=True, floating=False, verbose=2, scoring='r2', cv=10)
    selector.fit(X, y)
    selected_features = list(selector.k_feature_idx_)
    
    # Extract file metadata
    file_id = re.findall("\\d+", os.path.basename(pin_file))
    df_h[f"s{file_id[0]}_{file_id[1]}_{file_id[2]}"] = selected_features

# Save selected features to CSV
output_path = os.path.join(data_dir, "selected_features.csv")
df_h.to_csv(output_path, header=True)
print("Feature selection complete. Results saved at:", output_path)