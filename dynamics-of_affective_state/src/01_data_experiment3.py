import pandas as pd
import numpy as np
import os

def generate_data(df, data_dir, file_name):
    """
    Augments a DataFrame with skill level and math anxiety level columns, and saves it as a JSON file.

    Args:
        df: The input DataFrame.
        data_dir: The directory to save the JSON file.
        file_name: The name of the JSON file.

    Returns:
        None

    """

    # Create new columns
    df["skill_level"] = np.random.randint(1, 6, size=len(df))
    df["math_anxiety_level"] = np.random.randint(1, 6, size=len(df))

    file_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_path):
        # Record the csv data
        df.to_json(file_path, index=False)
        print(f"File created successfully: {file_path}")

    else:
        print(f"File already exists: {file_path}")


DATA_DIR = "data"
DATA = pd.read_json(f"{DATA_DIR}/CoMTA_dataset.json")
generate_data(DATA, DATA_DIR, "CoMTA_dataset3.json")
