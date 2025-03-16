import os
import pandas as pd
from groq import Groq
from emo_pred import ExperimentRunner
from dotenv import load_dotenv

DATA_DIR = "../data"
OUT_DIR = "../data/output"

load_dotenv(dotenv_path="../../.env")
API_KEY = os.getenv("GROQ_API")

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

##########################################################################################################
##
# Experiment 1
##
##########################################################################################################

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
experiment1_1.run_experiment()

experiment1_2 = ExperimentRunner(
    DATA[:1], MODEL, CLIENT, PROMPT_12, STATES, "experiment_1_model2", model_1=False
)
# experiment1_2.run_experiment()

##########################################################################################################
##
# Experiment 2
##
##########################################################################################################

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
    DATA[:2], MODEL, CLIENT, PROMPT_21, STATES, "experiment_2_model1", model_1=True
)
# experiment2_1.run_experiment()

experiment2_2 = ExperimentRunner(
    DATA[:1], MODEL, CLIENT, PROMPT_22, STATES, "experiment_2_model2", model_1=False
)
# experiment2_1.run_experiment()

##########################################################################################################
##
# Experiment 3
##
##########################################################################################################

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
    DATA3[:3], MODEL, CLIENT, PROMPT_31, STATES, "experiment_3_model1_", model_1=True
)
# experiment3_1.run_experiment()

experiment3_2 = ExperimentRunner(
    DATA3[:1], MODEL, CLIENT, PROMPT_32, STATES, "experiment_3_model2", model_1=False
)
# experiment3_2.run_experiment()
