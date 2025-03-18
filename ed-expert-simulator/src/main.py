import json
import os
import random
import numpy as np
import pandas as pd
import requests

# Define constants
URL = "https://raw.githubusercontent.com/joyheyueya/declarative-math-word-problem/refs/heads/main/algebra222.csv"
OUT_DIR = "../data"
FILE_NAME = "algebra222.csv"
DF_ALGEBRA = pd.read_csv(URL)
FILE_PATH = os.path.join(OUT_DIR, FILE_NAME)

# Create the data directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)


def generate_personas():

    level_1 = random.randint(1, 5)
    level_2 = random.randint(1, 5)

    return {
        "skill_level_1": level_1,
        "skill_level_2": level_2,
        "total_expertise": level_1 + level_2,
    }


def personas_expertise_level(students):
    # Calculate the median expertise level
    total_expertise_level = [s["total_expertise"] for s in students]
    median_expertise_level = np.median(total_expertise_level)

    # Assign expertise level based on the total expertise median level
    for i in students:
        if i["total_expertise"] <= median_expertise_level:
            i["expertise"] = "low-expertise"

        else:
            i["expertise"] = "high-expertise"
    return students


def download_file(url, file_path):
    """
    Downloads a file from a given URL if it is not already present.
    """
    if not os.path.exists(file_path):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()  # Raise an error for failed requests

            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"File downloaded successfully: {file_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
    else:
        print(f"File already exists: {file_path}")


def read_csv(file_path: str):
    """Reads and parses a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)  # Parse JSON data
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


# Define the full file path
file_path = os.path.join(OUT_DIR, FILE_NAME)

# Download the file if it doesn't exist
download_file(URL, file_path)

# Read and parse the JSON file
csv_data = read_csv(file_path)

# Print the JSON content (formatted)
if csv_data:
    print("JSON Data Loaded Successfully:")
    print(json.dumps(csv_data, indent=4))


num = 200
personas_list = [generate_personas() for i in range(num)]
personas_list = personas_expertise_level(personas_list)
df_persona = pd.DataFrame(personas_list)
