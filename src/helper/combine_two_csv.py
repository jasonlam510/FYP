# Combine two csv files
import pandas as pd

def combine_two_csv(path_1: str, path_2: str, save_path: str):
    df1 = pd.read_csv(path_1)
    print(f"Number of rows in df1: {len(df1)}")

    df2 = pd.read_csv(path_2)
    print(f"Number of rows in df2: {len(df2)}")

    df = pd.concat([df1, df2])
    df.to_csv(save_path, index=False)
    print(f"Combined csv files saved to {save_path}")
    print(f"Number of rows in the combined dataframe: {len(df)}")

if __name__ == "__main__":
    import os, sys
    from pathlib import Path
    
    project_root = Path.cwd() # Get the current directory (FYP/kaggle)
    sys.path.append(str(project_root))

    path1 = project_root / 'data' / 'processed' / 'kaggle' / 'kaggle_annotated_cleaned.csv'
    path2 = project_root / 'data' / 'processed' / 'kaggle' / 'llm_missing_features.csv'
    save_path = project_root / 'data' / 'processed' / 'kaggle' / 'feature_extracted_data.csv'

    combine_two_csv(path1, path2, save_path)



