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
parser = argparse.ArgumentParser(description="Perform evaluation (accuracy)")
parser.add_argument(
    "--inputfile",
    required=True,
    help="Path to the input .csv file containing evaluation samples"
)
parser.add_argument(
    "--outputfile",
    required=True,
    help="Path to the output .csv file containing evaluations (accuracy)"
)
parser.add_argument(
    "--inputprompts",
    required=True,
    help="Absolute file path of the JSON file containing prompts and prompt templates used for inference"
)
parser.add_argument(
    "--inputcred",
    required=True,
    help="Absolute file path of the .env file containing OPENAI_API_KEY"
)
args = parser.parse_args()

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------Load OpenAI credentials from the provided .env file----------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
load_dotenv(Path(args.inputcred))

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------Initialize OpenAI client-------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
client = OpenAI()

### --------------------------------------------------------------------------------------------------------- ###
### ----------------Create local output directory to store dataframe containing evaluations------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
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
        image_url
):
    """
    This function calls the OpenAI API.
    This leverages code from https://platform.openai.com/docs/guides/images-vision?api-mode=responses#analyze-images
            Parameters:
                    developer_prompt (str): Developer prompt passed to OpenAI API.
                    user_prompt (str): User prompt passed to OpenAI API.
                    image_url (str | None): URL to meme.

            Returns:
                    tuple containing

                            - openai_response_json (str): Response from calling the OpenAI API, in JSON string format.
                            - openai_response_text (str): Text content of the response.
    """
    response = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        reasoning={"effort": "high"},
        input=[{"role": "developer", "content": developer_prompt}, {"role": "user", "content": [{"type": "input_text", "text": user_prompt}, {"type": "input_image", "image_url": image_url, "detail": "high"}]}],
        max_output_tokens=32768,
    )
    openai_response_json = response.model_dump_json(indent=2)
    openai_response_text = getattr(response, "output_text", "") or ""
    return openai_response_json, openai_response_text


### --------------------------------------------------------------------------------------------------------- ###
### -------------------------Define main function that applies the inference function------------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
def main():
    memes = pd.read_csv(
        filepath_or_buffer=args.inputfile,
        encoding="utf-8"
    )
    with open(file=args.inputprompts, mode="r", encoding="utf-8") as file:
        prompts = json.load(file)
    for i in range(12):
        for run in range(5):
            memes[
                [
                    f"openai_response_json_quintuplet_{i}_run_{run+1}",
                    f"assess_quintuplet_{i}_run_{run+1}"
                ]
            ] = memes.p_apply(
                lambda row: inference(
                    developer_prompt=prompts["system_prompt"],
                    user_prompt=row[f"viewpoint_{i}"] + "\n" +
                                row[f"target_{i}"] + "\n" +
                                row[f"aspect_{i}"] + "\n" +
                                row[f"opinion_{i}"] + "\n" +
                                row[f"sentiment_{i}"],
                    image_url=row["meme_s3"],
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
