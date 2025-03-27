import re
import random
import pandas as pd
import groq


class ExperimentRunner:
    def __init__(
        self, df, model, client, prompt_template, states, experiment_name, model_1=True
    ):
        """
        Initializes the experiment runner.

        Args:
            df: Dataframe containing student conversations.
            model: LLM model name.
            client: API client for model inference.
            prompt_template: Template string for generating prompts.
            states: List of possible emotional states.
            experiment_name: Name of the experiment.
            model_1: Boolean flag to determine which model is used (1 or 2).
        """
        self.df = df
        self.model = model
        self.client = client
        self.prompt_template = prompt_template
        self.states = states
        self.experiment_name = experiment_name
        self.model_1 = model_1

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
                max_tokens=512,  # The maximum number of tokens to generate.
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

    def _generate_prompt(
        self,
        math_level,
        skill_level,
        math_anxiety_level,
        student_response,
        tutor_response,
        transcript_pairs,
        current_state=None,
    ):
        """
        Generates a prompt for model inference.

        Args:
            student_response: Student's message.
            tutor_response: Tutor's message.
            transcript_pairs: Previous conversation history.
            current_state: The student's emotional state.
            Formatted prompt string.
        """
        if self.model_1:
            return self.prompt_template.format(
                math_level=math_level,
                skill_level=skill_level,
                math_anxiety_level=math_anxiety_level,
                s_response=student_response,
                t_response=tutor_response,
                states=self.states,
                current_state=(
                    current_state if current_state else random.choice(self.states)
                ),
            )
        else:
            # For model_2, format the transcript pairs
            transcript_pairs.append(
                {"student": student_response, "tutor": tutor_response}
            )
            formatted_transcript = "\n".join(
                [
                    f"  Student: {pair['student']}\n  Tutor: {pair['tutor']}"
                    for pair in transcript_pairs
                ]
            )

            return self.prompt_template.format(
                math_level=math_level,
                skill_level=skill_level,
                math_anxiety_level=math_anxiety_level,
                transcript=formatted_transcript,
                states=self.states,
                current_state=(
                    current_state if current_state else random.choice(self.states)
                ),
            )

    def run_experiment(self, times=1):
        """
        Executes the experiment by iterating through the dataset and making predictions.

        Saves results to a CSV file.
        """
        result = []
        for t in range(times): # How many times will it run
            for c, row in self.df.iterrows():
                conversation = row["data"]
                math_level = row.get("math_level", None)
                skill_level = row.get("skill_level", None)
                math_anxiety_level = row.get("math_anxiety_level", None)
                previous_state = ""
                transcript_pairs = []
                time_step = 1
                for i in range(0, len(conversation), 2):
                    student_response = (
                        conversation[i]["content"]
                        if conversation[i]["role"] == "user"
                        else None
                    )

                    tutor_response = (
                        conversation[i + 1]["content"]
                        if i + 1 < len(conversation)
                        and conversation[i + 1]["role"] == "assistant"
                        else None
                    )
                    if student_response and tutor_response:
                        current_state = (
                            random.choice(self.states)
                            if not previous_state
                            else previous_state
                        )
                        # Add to transcript_pairs only if model_1 is not used
                        # if self.model_1 is False:
                        #     transcript_pairs.append(
                        #         {"student": student_response, "tutor": tutor_response}
                        #     )
                        
                        print(f"{'@'*70}\n{transcript_pairs}\n{'@'*70}")
                        # Generate prompt
                        prompt = self._generate_prompt(
                            math_level,
                            skill_level,
                            math_anxiety_level,
                            student_response,
                            tutor_response,
                            transcript_pairs,
                            current_state,
                        )

                        # Make prediction
                        llm_response = self._call_groq_api(prompt)
                        predicted_emotion_match = re.findall(r"\[(.*?)\]", llm_response)
                        predicted_emotion = (
                            predicted_emotion_match[0] if predicted_emotion_match else None
                        )

                        # Store results
                        result.append(
                            {
                                "t": t, # t times running the prompt
                                "student_id": c + 1,
                                "time_step": time_step,
                                "student_response": student_response,
                                "tutor_response": tutor_response,
                                "previous_state": previous_state,
                                "current_state": current_state,
                                "next_state": predicted_emotion,
                                "math_level": math_level,
                                "skill_level": skill_level,
                                "math_anxiety_level": math_anxiety_level,
                                "prompt": prompt,
                                "llm_response": llm_response,
                            }
                        )

                        # Update state
                        previous_state = predicted_emotion

                        # Debugging output
                        print(f"{t}\nPrompt:\n{prompt}\n")
                        print(
                            f"\n-----------Student {c + 1} | time step {time_step}-----------"
                        )
                        print(f"LLM Response: {llm_response}\n")
                        print(f"Predicted Emotion: {predicted_emotion}\n")
                    time_step += 1
        # Convert to DataFrame and save to CSV
        df_result = pd.DataFrame(result)
        df_result.to_csv(f"../data/output/{self.experiment_name}.csv", index=False)

        print(f"Experiment {self.experiment_name} completed and saved!")
