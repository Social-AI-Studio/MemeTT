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
    split_factor=8
)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------Specify required command line arguments-------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description="Perform evaluations")
parser.add_argument(
    "--inputfile",
    required=True,
    help="Absolute file path of the .csv file containing clusters"
)
parser.add_argument(
    "--clustercol",
    required=True,
    help="Name of column where cluster labels or narratives are stored"
)
parser.add_argument(
    "--outputfile",
    required=True,
    help="Path to the output containing evaluations of clusters"
)
parser.add_argument(
    "--inputprompts",
    required=True,
    help="Absolute file path of the JSON file containing MABSA prompts and prompt templates used for inference"
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
### -----------------Create local output directory to store dataframe containing evaluations----------------- ###
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
        image_urls
):
    """
    This function calls the OpenAI API.
    This leverages code from https://platform.openai.com/docs/guides/images-vision?api-mode=responses#analyze-images
            Parameters:
                    developer_prompt (str): Developer prompt passed to OpenAI API.
                    image_urls (list[str]): List of URLs to memes.

            Returns:
                    tuple containing

                            - openai_response_json (str): Response from calling the OpenAI API, in JSON string format.
                            - openai_response_text (str): Text content of the response.
    """
    user_content = [
        item
        for i, url in enumerate(image_urls, start=1)
        for item in (
            {"type": "input_text", "text": f"This is meme {i}."},
            {"type": "input_image", "image_url": url, "detail": "high"},
        )
    ]

    response = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        reasoning={"effort": "high"},
        input=[
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": user_content},
        ],
        max_output_tokens=128000,
    )
    openai_response_json = response.model_dump_json(indent=2)
    openai_response_text = getattr(response, "output_text", "")
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
    unique_narratives = (
        memes.groupby(args.clustercol, as_index=False)
            .agg(urls=("meme_s3", lambda s: list(pd.unique(s.dropna()))))
            .assign(meme_count=lambda d: d["urls"].str.len())
            .loc[lambda d: d["meme_count"].gt(1)]
            .reset_index(drop=True)
    )
    for i in range(1, 4):
        unique_narratives[
            [f"openai_response_json_cluster_quality_{i}", f"cluster_quality_{i}"]
        ] = unique_narratives.p_apply(
            lambda row: inference(
                developer_prompt=prompts["system_prompt"].format(n=row.meme_count),
                image_urls=row.urls,
            ),
            axis=1,
            result_type="expand"
        )
    unique_narratives.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfile)), os.path.basename(args.outputfile)),
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
