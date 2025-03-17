import os
import json
import random


def generate_data(input_file: str, output_file: str):
    """Augments a dataset with skill level and math anxiety level entry, and saves it as a JSON file.

    Args:
        input_file (str): input data file path
        output_file (str): output data file
    """
    # If output file doesn't exist, generate new json file
    if not os.path.exists(output_file):
        with open(input_file, "r") as file:
            data = json.load(file)

            # Add the new keys with random values between 1 and 5
            for entry in data:
                entry["skill_level"] = random.randint(1, 5)
                entry["math_anxiety_level"] = random.randint(1, 5)

        # Save the updated JSON data back to the file
        with open(output_file, "w") as file:
            json.dump(data, file, indent=4)

        print(f"File created successfully: {output_file}")
    else:
        print(f"File already exists: {output_file}")


INPUT_DATA = "../data/CoMTA_dataset.json"
OUTPUT_DATA = "../data/CoMTA_dataset3.json"
generate_data(INPUT_DATA, OUTPUT_DATA)
