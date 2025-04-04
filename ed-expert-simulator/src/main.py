import os
import random
import numpy as np
import pandas as pd
from groq import Groq
import dotenv import load_dotenv

DATA_DIR = '../data'
OUTPUT_DIR = '../data/output'

dotenv_path = os.path.abspath("../../.env")
load_dotenv(dotenv_path=dotenv_path)

# Retrieve the API key
API_KEY = os.getenv("GROQ_API")

# Check if the API key is loaded
if not API_KEY:
    raise ValueError("API key is missing. Check your .env file and path.")

CLIENT = Groq(api_key=API_KEY)

LLAMA = "llama3-8b-8192"

# GEMMA = 

