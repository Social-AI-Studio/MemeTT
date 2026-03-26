### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import os
import json

import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from parallel_pandas import ParallelPandas
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

ParallelPandas.initialize(
    n_cpu=32,
    split_factor=2
)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------Specify required command line arguments-------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description="Perform evaluations (intra-cluster quality)")
parser.add_argument(
    "--inputcoherence",
    required=True,
    help="Path to the input .csv file containing narratives for evaluation"
)
parser.add_argument(
    "--inputrelevance",
    required=True,
    help="Path to the input .csv file containing narrative-meme pairs for evaluation"
)
parser.add_argument(
    "--outputcoherenceeval",
    required=True,
    help="Path to the output .csv file containing evaluations of narrative coherence"
)
parser.add_argument(
    "--outputrelevanceeval",
    required=True,
    help="Path to the output .csv file containing evaluations of narrative relevance"
)
parser.add_argument(
    "--inputprompts",
    required=True,
    help="Absolute file path of the JSON file containing prompts and prompt templates used for inference"
)
parser.add_argument(
    "--inputcred",
    required=True,
    help="Absolute file path of the .env file containing OpenAI key"
)
args = parser.parse_args()

### --------------------------------------------------------------------------------------------------------- ###
### --------------------------Set OPENAI_API_KEY environment variable for authentication--------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
load_dotenv(Path(args.inputcred))

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------Initialize OpenAI client-------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
client = OpenAI()

### --------------------------------------------------------------------------------------------------------- ###
### ----------------Create local output directory to store dataframe containing evaluations------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputcoherenceeval))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### -------------------Define a function to call the OpenAI API to generate evaluations---------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(max_attempt_number=100),
    reraise=False,
    retry_error_callback=lambda rs: (
        json.dumps({
            "status": "error",
            "etype": type(rs.outcome.exception()).__name__ if rs.outcome and rs.outcome.exception() else "None",
            "message": str(rs.outcome.exception()) if rs.outcome and rs.outcome.exception() else ""
        }),
        ""
    ),
)
def inference(
        developer_prompt,
        user_prompt,
        text_schema=None,
        image_url=None
):
    """
    This function calls the OpenAI API.
    This leverages code from https://platform.openai.com/docs/guides/images-vision?api-mode=responses#analyze-images
            Parameters:
                    developer_prompt (str): Developer prompt passed to OpenAI API.
                    user_prompt (str): User prompt passed to OpenAI API.
                    text_schema (str | None): Text schema which the output must conform to.
                    image_url (str | None): URL to memes.

            Returns:
                    tuple containing

                            - openai_response_json (str): Response from calling the OpenAI API, in JSON string format.
                            - openai_response_text (str): Text content of the response.
    """
    if not image_url:
        response = client.responses.create(
            model="gpt-5-mini-2025-08-07",
            reasoning={"effort": "high"},
            input=[
                {"role": "developer", "content": developer_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=32768,
        )
    else:
        response = client.responses.create(
            model="gpt-5-mini-2025-08-07",
            reasoning={"effort": "high"},
            input=[{"role": "developer", "content": developer_prompt}, {"role": "user", "content": [{"type": "input_text", "text": user_prompt}, {"type": "input_image", "image_url": image_url, "detail": "high"}]}],
            text=text_schema,
            max_output_tokens=32768,
        )
    openai_response_json = response.model_dump_json(indent=2)
    openai_response_text = getattr(response, "output_text", "") or ""
    return openai_response_json, openai_response_text


### --------------------------------------------------------------------------------------------------------- ###
### -------------------------Define main function that applies the inference function------------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
def main():
    coherence_eval = pd.read_csv(args.inputcoherence, encoding="utf-8")
    relevance_eval = pd.read_csv(args.inputrelevance, encoding="utf-8")
    with open(file=args.inputprompts, mode="r", encoding="utf-8") as file:
        prompts = json.load(file)
    for run in range(5):
        coherence_eval[[f"openai_response_json_coherence_{run}", f"coherence_answer_{run}"]] = coherence_eval.p_apply(lambda row: inference(developer_prompt=prompts["system_prompt_coherence"], user_prompt=prompts["user_prompt"].format(narrative=row.narrative, target=row.target_cluster_label, aspect=row.aspect_cluster_label_final, opinion=row.opinion_cluster_label_final, sentiment=row.sentiment)), axis=1, result_type="expand")
        relevance_eval[[f"openai_response_json_relevance_{run}", f"relevance_answer_{run}"]] = relevance_eval.p_apply(
            lambda row: inference(
                developer_prompt=prompts["system_prompt_relevance"],
                user_prompt=prompts["user_prompt"].format(
                    narrative=row.narrative,
                    target=row.target_cluster_label,
                    aspect=row.aspect_cluster_label_final,
                    opinion=row.opinion_cluster_label_final,
                    sentiment=row.sentiment
                ),
                image_url=row.meme_s3,
                text_schema={
                    "format": {
                        "type": "json_schema",
                        "name": "relevance_boolean",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "Q1": {"type": "boolean"},
                                "Q2": {"type": "boolean"},
                                "Q3": {"type": "boolean"}
                            },
                            "required": ["Q1", "Q2", "Q3"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            ),
            axis=1,
            result_type="expand"
        )
    coherence_eval.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputcoherenceeval)), os.path.basename(args.outputcoherenceeval)),
        index=False,
        encoding="utf-8"
    )
    relevance_eval.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputrelevanceeval)), os.path.basename(args.outputrelevanceeval)),
        index=False,
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
