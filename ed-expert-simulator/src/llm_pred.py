import json
import os
import random
import groq
import numpy as np
import pandas as pd
import requests

class ExperimentRunner:
    def __init__(
        self, df, client, model, prompt_template, experiment_name):
        self.df = df
        self.client = client
        self.model = model 
        self.prompt_template = prompt_template
        self.experiment_name = experiment_name
    
    
    def _call_groq_api(self, prompt):
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,  # The language model which will generate the completion.
                temperature=0,  # Controls randomness: lowering results in less random completions.
                max_tokens=256,  # The maximum number of tokens to generate.
                top_p=1,  # Controls diversity via nucleus sampling.
                stop=None,
                stream=False,  # If set, partial message deltas will be sent
            )
        except groq.APIConnectionError as e:
            print(
                "The server could not be reached"
            )  # an underlying Exception, likely raised within httpx.
            print(e.__cause__)
        except groq.RateLimitError:
            print(
                f"A 429 status code was received, we should back off a bit.\n{e.response}"
            )
        except groq.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
        return response.choices[0].message.content
    
    def _generate_prompt(self):
        pass
    
    def run_llm_prediction(self, times):
        pass