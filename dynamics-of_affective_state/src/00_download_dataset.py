import json
import os
import requests

# Define constants
OUT_DIR = "../data"
URL = "https://raw.githubusercontent.com/Khan/tutoring-accuracy-dataset/refs/heads/main/CoMTA_dataset.json"
FILE_NAME = "CoMTA_dataset.json"  # Define a file name

# Create the data directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)


def download_file(url, file_path):
    """
    Downloads a file from a given URL if it is not already present.
    """
    if not os.path.exists(file_path):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for failed requests

            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"File downloaded successfully: {file_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
    else:
        print(f"File already exists: {file_path}")


def read_json(file_path: str):
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
json_data = read_json(file_path)

# Print the JSON content (formatted)
if json_data:
    print("JSON Data Loaded Successfully:")
    print(json.dumps(json_data, indent=4))
