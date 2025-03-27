import os
import pandas as pd
from groq import Groq
from emo_pred import ExperimentRunner
from dotenv import load_dotenv

DATA_DIR = "../data"
OUT_DIR = "../data/output"

dotenv_path = os.path.abspath("../../.env")
load_dotenv(dotenv_path=dotenv_path)

# Retrieve the API key
API_KEY = os.getenv("GROQ_API")

# Check if the API key is loaded
if not API_KEY:
    raise ValueError("API key is missing. Check your .env file and path.")

CLIENT = Groq(api_key=API_KEY)

MODEL = "llama3-8b-8192"
DATA = pd.read_json(f"{DATA_DIR}/CoMTA_dataset.json")
DATA3 = pd.read_json(f"{DATA_DIR}/CoMTA_dataset3.json")
STATES = [
    "boredom",
    "engagement",
    "confusion",
    "frustration",
    "delight",
    "surprise",
    "neutral",
]

####################################################################################################
##
# Experiment 1
##
####################################################################################################

PROMPT_11 = """
A student is using an intelligent tutoring system on math problem solving.
The system adjusts its guidance based on the student's responses.

Here's the dialog: 
    - Student: {s_response}
    - Tutor: {t_response}

Based on their transcript, which emotional state is the student most likely to feel next?
Choose only one from the following options: {states}
    - Format: Write the answer inside square brackets.
    - Reasoning: Briefly explain why this emotional state is the most likely.
"""

PROMPT_12 = """
A student is using an intelligent tutoring system on math problem solving.
The system adjusts its guidance based on the student's responses.

Here's the dialog: \n{transcript}\n

Based on their transcript, which emotional state is the student most likely to feel next?
Choose only one from the following options: {states}
- Format: Write the answer inside square brackets.
- Reasoning: Briefly explain why this emotional state is the most likely.
"""

# Run experiment 1
experiment1_1 = ExperimentRunner(
    DATA, MODEL, CLIENT, PROMPT_11, STATES, "experiment_1_model1", model_1=True
)
# experiment1_1.run_experiment() # replaced by experiment 2

experiment1_2 = ExperimentRunner(
    DATA, MODEL, CLIENT, PROMPT_12, STATES, "experiment_1_model2", model_1=False
)
# experiment1_2.run_experiment()  # replaced by experiment 2

##############################################################################################
##
# Experiment 2
##
##############################################################################################

PROMPT_21 = """
A student is using an intelligent tutoring system on math problem solving.
The system adjusts its guidance based on the student's responses and their current emotion state.\n

Here’s the student's:
    - Current emotional state: [{current_state}]
    - Dialogue:
        Student: {s_response}
        Tutor: {t_response}

Based on the conversation, predict the student's most likely next emotional state.
Choose one from the following options: {states}
    - Format: Write the answer inside square brackets (e.g., [Emotion]).
    - Reasoning: Briefly explain why this emotional state is the most likely.
"""

PROMPT_22 = """
The system adapts its guidance based on the student's responses and emotional state.
Here’s the student's:
    - Current emotional state: [{current_state}]
    - Dialogue: \n{transcript}\n

Based on the conversation, predict the student's most likely next emotional state.
Choose one from the following options: {states}
    - Format: Write the answer inside square brackets (e.g., [Emotion]).
    - Reasoning: Briefly explain why this emotional state is the most likely.
"""

experiment2_1 = ExperimentRunner(
    DATA3, MODEL, CLIENT, PROMPT_21, STATES, "experiment_2_model1_5x", model_1=True
)
# experiment2_1.run_experiment(2) # succeeded

experiment2_2 = ExperimentRunner(
    DATA3, MODEL, CLIENT, PROMPT_22, STATES, "experiment_2_model2_2x", model_1=False
)
# experiment2_2.run_experiment(2)

####################################################################################################
##
# Experiment 3
##
####################################################################################################

PROMPT_31 = """
Here's a student who is interacting with an intelligent tutoring system on a {math_level} problem-solving task
Their skill levels (each skill is rated on a scale from 1 to 5):
    1. Problem-solving skill level: {skill_level}
    2. Math anxiety level: {math_anxiety_level}

The system adapts its guidance based on the student's responses and emotional state.
Here’s the student's:
    - Current emotional state: [{current_state}]
    - Dialogue:
        Student: {s_response}
        Tutor: {t_response}

Based on the conversation, predict the student's most likely next emotional state.
Choose one from the following options: {states}
    - Format: Write the answer inside square brackets (e.g., [Emotion]).
    - Reasoning: Briefly explain why this emotional state is the most likely.

"""
PROMPT_32 = """
Here's a student who is interacting with an intelligent tutoring system on a {math_level} problem-solving task
Their skill levels (each skill is rated on a scale from 1 to 5):
    1. Problem-solving skill level: {skill_level}
    2. Math anxiety level: {math_anxiety_level}

The system adapts its guidance based on the student's responses and emotional state.
Here’s the student's:
    - Current emotional state: [{current_state}]
    - Dialogue: \n{transcript}\n

Based on the conversation, predict the student's most likely next emotional state.
Choose one from the following options: {states}
    - Format: Write the answer inside square brackets (e.g., [Emotion]).
    - Reasoning: Briefly explain why this emotional state is the most likely.
"""

# Run experiment 3
experiment3_1 = ExperimentRunner(
    DATA3, MODEL, CLIENT, PROMPT_31, STATES, "experiment_3_model1_2x", model_1=True
)
# experiment3_1.run_experiment(2) # succeeded

experiment3_2 = ExperimentRunner(
    DATA3, MODEL, CLIENT, PROMPT_32, STATES, "experiment_3_model2_2x", model_1=False
)
experiment3_2.run_experiment(2)
