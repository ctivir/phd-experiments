import os
import random
import numpy as np
import pandas as pd
import requests


class PersonaManager:
    """
    Handles downloading, processing, and analyzing personas from a CSV file.
    """

    def __init__(self, url, out_dir="../data", file_name="algebra222.csv"):
        self.url = url
        self.out_dir = out_dir
        self.file_name = file_name
        self.file_path = os.path.join(out_dir, file_name)

        # Ensure the data directory exists
        os.makedirs(self.out_dir, exist_ok=True)

    def download_file(self):
        """
        Downloads the CSV file if it doesn't already exist.
        """
        if os.path.exists(self.file_path):
            print(f"File already exists: {self.file_path}")
            return

        try:
            response = requests.get(self.url, timeout=60)
            response.raise_for_status()

            with open(self.file_path, "wb") as file:
                file.write(response.content)

            print(f"File downloaded successfully: {self.file_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")

    @staticmethod
    def generate_persona():
        """
        Generates a single persona with random skill levels.

        """
        level_1 = random.randint(1, 5)
        level_2 = random.randint(1, 5)
        return {
            "level1": level_1,
            "level2": level_2,
            "total_expertise": level_1 + level_2,
        }

    def generate_personas(self, num_personas=200):
        """
        Generates a list of personas with skill levels.

        Args:
            num_personas (int): Number of personas to generate.

        Returns:
            list: A list of persona dictionaries.
        """
        return [self.generate_persona() for _ in range(num_personas)]

    @staticmethod
    def assign_expertise_and_conditions(personas, num=None):
        """
        Assigns expertise levels based on median total expertise, then assigns
        expertise reversal and variability conditions.

        Args:
            personas (list or DataFrame): List of persona dictionaries.
            num (int): Number of personas to assign expertise reversal conditions.

        Returns:
            DataFrame: Personas with assigned expertise levels and conditions.
        """
        if not isinstance(personas, pd.DataFrame):
            personas = pd.DataFrame(personas)

        # Assign expertise levels based on median total expertise
        median_expertise = np.median(personas["total_expertise"])
        personas["expertise"] = personas["total_expertise"].apply(
            lambda x: "low-expertise" if x <= median_expertise else "high-expertise"
        )
        print(median_expertise)

        # Ensure num is provided for expertise reversal assignment
        if num is None:
            raise ValueError("The 'num' parameter is required for expertise reversal.")
        print(num)
        
        personas = personas.sample(frac=1, random_state=42).reset_index(
            drop=True
        )  # Shuffle personas
        
        # Assign variability conditions
        variability_conditions = [
            "low-variability/practice",
            "low-variability/worked example",
            "high-variability/practice",
            "high-variability/worked example",
        ]
        personas["variability"] = [
            variability_conditions[i % 4] for i in range(len(personas))
        ]
        
        print(personas.head())
        
        low_condition = ["low-expertise/practice", "low-expertise/worked-example"]
        high_condition = ["high-expertise/practice", "high-expertise/worked-example"]

        # Split personas by expertise level
        low_group = personas[personas["expertise"] == "low-expertise"]
        high_group = personas[personas["expertise"] == "high-expertise"]

        # Ensure each group has enough personas
        if len(low_group) < num or len(high_group) < num:
            raise ValueError(
                "Not enough personas in one of the groups to match the specified size."
            )

        # Sample `num` personas from each group
        low_group = low_group.sample(n=num, random_state=1).reset_index(drop=True)
        high_group = high_group.sample(n=num, random_state=1).reset_index(drop=True)

        # Assign expertise reversal conditions
        low_group["expertise_reversal"] = [low_condition[i % 2] for i in range(num)]
        high_group["expertise_reversal"] = [high_condition[i % 2] for i in range(num)]

        # Merge assigned personas and shuffle again
        assigned_personas = (
            pd.concat([low_group, high_group])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        print(assigned_personas.head())

        return assigned_personas.sample(frac=1, random_state=42).reset_index(drop=True)

    def save_personas_to_csv(self, personas, file_name=None):
            """
            Saves the personas DataFrame to a CSV file.

            Args:
                personas (DataFrame or list): The persona data to be saved.
                file_name (str, optional): Custom file name for the CSV.

            Returns:
                str: Path to the saved CSV file.
            """
            if not isinstance(personas, pd.DataFrame):
                personas = pd.DataFrame(personas)

            # Define file path
            file_name = file_name or self.file_name
            file_path = os.path.join(self.out_dir, file_name)

            # Save to CSV
            try:
                personas.to_csv(file_path, index=False)
                print(f"Personas saved successfully to {file_path}")
                return file_path
            except Exception as e:
                print(f"Error saving personas to CSV: {e}")
                return None
        
        
# Usage
URL = "https://raw.githubusercontent.com/joyheyueya/declarative-math-word-problem/refs/heads/main/algebra222.csv"

# Create an instance of PersonaManager
persona_manager = PersonaManager(URL)

# Download and load CSV data
persona_manager.download_file()

# Generate and process personas
personas_list = persona_manager.generate_personas(1000)
personas_list = persona_manager.assign_expertise_and_conditions(personas_list, 100)

# Save personas to a CSV file
persona_manager.save_personas_to_csv(personas_list, "personas.csv")
