### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import ast
import os
import json

import pandas as pd

from dotenv import load_dotenv
from mistralai import Mistral
from parallel_pandas import ParallelPandas
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

ParallelPandas.initialize(
    n_cpu=64,
    split_factor=2
)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------Specify required command line arguments-------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description="Perform MABSA")
parser.add_argument(
    "--inputfile",
    required=True,
    help="Absolute file path of the .csv file containing meme URLs, OCR responses and meme texts, recognized celebrities and their information for the combined dataset"
)
parser.add_argument(
    "--outputfile",
    required=True,
    help="Path to the output .csv file containing MABSA results"
)
parser.add_argument(
    "--inputprompts",
    required=True,
    help="Absolute file path of the JSON file containing MABSA prompts and prompt templates used for inference"
)
parser.add_argument(
    "--inputcred",
    required=True,
    help="Absolute file path of the .env file containing Mistral AI API key"
)
args = parser.parse_args()

### --------------------------------------------------------------------------------------------------------- ###
### -----------------------------Load Mistral AI credentials from the provided .env file--------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
load_dotenv(Path(args.inputcred))

### --------------------------------------------------------------------------------------------------------- ###
### -----------------------------------------Initialize Mistral AI client------------------------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

### --------------------------------------------------------------------------------------------------------- ###
### ----------------Create local output directory to store dataframe containing MABSA inferences------------- ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### ----------------Define a function to call the Mistral AI API to perform MABSA inference------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(max_attempt_number=100)
)
def inference(
        image_url,
        system_prompt,
        user_prompt
):
    """
    This function calls the Mistral AI API.
    This leverages code from https://docs.mistral.ai/capabilities/vision/

            Parameters:
                    image_url (str): URL to meme.
                    system_prompt (str): System prompt passed to Mistral AI API.
                    user_prompt (str): User prompt passed to Mistral AI API.

            Returns:
                    tuple containing

                            - mistral_response_json (str): Response from calling the Mistral AI API, in JSON string format.
                            - mistral_response_text (str): Text content of the response.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": image_url
                }
            ]
        }
    ]
    mistral_response = client.chat.complete(
        model="mistral-small-2503",
        max_tokens=8192,
        messages=messages,
        random_seed=42,
        temperature=0
    )
    mistral_response_json = mistral_response.model_dump_json()
    mistral_response_text = mistral_response.choices[0].message.content
    return mistral_response_json, mistral_response_text


### --------------------------------------------------------------------------------------------------------- ###
### -------------------------Define main function that applies the inference function------------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
def main():
    memes = pd.read_csv(
        filepath_or_buffer=args.inputfile,
        encoding="utf-8"
    )
    memes.celebrities = memes.celebrities.apply(lambda x: ast.literal_eval(x) if not pd.isna(x) and x is not None else None)
    with open(args.inputprompts, mode="r", encoding="utf-8") as file:
        prompts = json.load(file)
    memes[["mistral_small_2503_response_json", "mistral_small_2503_response_text"]] = memes.p_apply(
        lambda row: inference(
            image_url=row["meme_bboxes_drawn_s3"],
            system_prompt=prompts["system_prompt"],
            user_prompt=prompts["user_prompt"] if not (isinstance(row["celebrities"], list) and row["celebrities"]) else prompts["prefix"].format(information=" ".join([prompts["identities"].format(color=celebrity[1], name=celebrity[0], description=celebrity[2]) for celebrity in row["celebrities"]])) + prompts["user_prompt"]
        ),
        axis=1,
        result_type="expand"
    )
    memes.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfile)), os.path.basename(args.outputfile)),
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
