import os
import json
import re
import pandas as pd
import groq
from dotenv import load_dotenv
from groq import Groq


def _call_groq_api(prompt, client, model):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,  # The language model which will generate the completion.
            temperature=0.1,  # Controls randomness: lowering results in less random completions.
            max_tokens=256,  # The maximum number of tokens to generate.
            top_p=1,  # Controls diversity via nucleus sampling.
            stop=None,
            stream=False,  # If set, partial message deltas will be sent
        )
    except groq.APIConnectionError as e:
        # an underlying Exception, likely raised within httpx.
        print("The server could not be reached")
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


def generate_tutor_response(json_data, prompt_template, client, model):
    """Generates a tutor response base on a given prompt 

    Args:
        json_data (df): the input json file 
        prompt_template (str): _description_
        client (str): _description_
        model (str): _description_

    Returns:
        json: return a json file
    """
    # Extract the conversation history
    for _, row in json_data.iterrows():
        conversation = row["data"]

        i = 0  # Track position manually to adjust after insertions
        while i < len(conversation):
            if (
                conversation[i]["role"] == "user"  # user message
                and i + 1 < len(conversation)
                and conversation[i + 1]["role"]
                == "assistant"  # follows by assistant message
            ):
                student_response = conversation[i]["content"]
                tutor_response = conversation[i + 1]["content"]

                # Add the instruction for the LLM
                prompt = prompt_template.format(
                    s_response=student_response, t_response=tutor_response
                )
                # Extract the generated tutor response
                response = _call_groq_api(prompt, client, model)
                extracted_new_response = re.findall(r"\[(.*?)\]", response)

                new_tutor_response_content = (
                    extracted_new_response[0] if extracted_new_response else ""
                )

                new_response = {
                    "role": "tutor_neutral",
                    "content": new_tutor_response_content,
                }
                conversation.insert(i + 2, new_response)
                i += 3  # Move the index forward by 3 (user -> assistant -> tutor_neutral)

                # Debugging output
                print(
                    f"{'-'*100}\n{prompt}\nResponse: {response}\nNew tutor response: "
                    f"{extracted_new_response}\n{new_response}"
                )

            else:
                i += 1  # Move to the next message
        # Append the new response to the conversation history
    return json_data


# Example usage
if __name__ == "__main__":
    PROMPT = """
    As an expert reviewer evaluating a tutor's response in a student-tutor 
    conversation, analyze the interaction below:
        - student: {s_response}
        - tutor: {t_response}
    Instruction: Generate an Improved Tutor Response
        1. Rewrite the tutor's response in neutral tone should be informative but no warmth or encouragement.
        2. Ensure the new response maintains clarity and according to the student response tone.
        3. Provide a brief explanation of the changes made.
        4. Do not actually answer the math question; just rephrase the tutor's response.
    Format your response as follows:
    - Format: Write the improved tutor response inside square brackets.
    - Reasoning: Briefly explain what was improved and why
    """

    dotenv_path = os.path.abspath("../../.env")
    load_dotenv(dotenv_path=dotenv_path)

    # Retrieve the API key
    API_KEY = os.getenv("GROQ_API")

    # Check if the API key is loaded
    if not API_KEY:
        raise ValueError("API key is missing. Check your .env file and path.")

    CLIENT = Groq(api_key=API_KEY)

    MODEL = "llama3-8b-8192"

    DATA = pd.read_json("../data/CoMTA_dataset3.json")
    OUTPUT_DATA = "../data/CoMTA_dataset_neutral_tutor.json" #CoMTA_dataset_impatient_tutor.json

    # Generate a new tutor response
    new_tutor_response = generate_tutor_response(DATA, PROMPT, CLIENT, MODEL)

    # Convert the DataFrame to a list of dictionaries before serializing to JSON
    new_tutor_response_json = new_tutor_response.to_dict(orient="records")
    print(json.dumps(new_tutor_response_json, indent=4))

    # Save the JSON data to a file
    with open(OUTPUT_DATA, "w", encoding="utf-8") as file:
        json.dump(new_tutor_response_json, file, indent=4)
