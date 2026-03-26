### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import ast
import base64
import json
import os
import requests

import fireworks.client
import pandas as pd

from dotenv import load_dotenv
from parallel_pandas import ParallelPandas
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

ParallelPandas.initialize(n_cpu=32)

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
    help="Absolute file path of the .env file containing Fireworks API key"
)
args = parser.parse_args()

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------------Load Fireworks credentials from the provided .env file--------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
load_dotenv(Path(args.inputcred))

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------------Set Fireworks API key for the SDK----------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
fireworks.client.api_key = os.getenv("FIREWORKS_API_KEY")

### --------------------------------------------------------------------------------------------------------- ###
### ----------------Create local output directory to store dataframe containing MABSA inferences------------- ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### ---------------Define a function to call the FIREWORKS AI API to perform MABSA inference----------------- ###
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
    This function calls the Fireworks AI API.
    This leverages code from https://docs.fireworks.ai/guides/querying-vision-language-models

            Parameters:
                    image_url (str): URL to meme.
                    system_prompt (str): System prompt passed to Fireworks AI API.
                    user_prompt (str): User prompt passed to Fireworks AI API.

            Returns:
                    tuple containing

                            - qwen_response_json (str): Response from calling the Fireworks AI API, in JSON string format.
                            - qwen_response_text (str): Text content of the response.
    """
    r = requests.get(image_url)
    r.raise_for_status()
    content = r.content
    base64_bytes = base64.b64encode(content).decode("utf-8")
    qwen_response = fireworks.client.ChatCompletion.create(
        model="accounts/fireworks/models/qwen2p5-vl-32b-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_bytes}"
                        },
                    }
                ]
            }
        ],
        max_tokens=8192,
        temperature=0,
        top_k=1
    )
    qwen_response_json = qwen_response.model_dump_json()
    qwen_response_text = qwen_response.choices[0].message.content
    return qwen_response_json, qwen_response_text


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
    memes[["qwen2p5_vl_32b_instruct_response_json", "qwen2p5_vl_32b_instruct_basic_response_text"]] = memes.p_apply(
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
