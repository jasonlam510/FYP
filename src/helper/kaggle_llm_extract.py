import sys
import pandas as pd
import logging
from datetime import datetime
import os
from pathlib import Path

# Get the current directory (FYP/kaggle)
current_dir = Path.cwd()

# Go up one level to FYP
project_root = current_dir
# print(project_root)
# input()
sys.path.append(str(project_root))

RAW_FILE_PATH = project_root / 'data' / 'raw' / 'kaggle' / 'bbc_news.csv'
SAVE_PATH = project_root / 'data' / 'processed' / 'kaggle'

import pandas as pd

# Load data from CSV files
A = pd.read_csv(str(SAVE_PATH) + "/sentiment_inserted.csv")
B = pd.read_csv(str(SAVE_PATH) + "/kaggle_annotated.csv")

# Remove rows in B that don't exist in A based on the 'title' column
B_cleaned = B[B['title'].isin(A['title'])]

# Create a new DataFrame C which contains the rows in A but not in B
C = A[~A['title'].isin(B['title'])]

# Print the number of rows in B that don't exist in A
rows_in_B_not_in_A = B.shape[0] - B_cleaned.shape[0]
print(f"Number of rows in B that don't exist in A: {rows_in_B_not_in_A}")

# Save the cleaned B (B_cleaned) to a new CSV file
B_cleaned.to_csv(str(SAVE_PATH) + "/kaggle_annotated_cleaned.csv", index=False)

# Optionally, save C to a new CSV or display it
C.to_csv(str(SAVE_PATH) + "/missing_features.csv", index=False)


