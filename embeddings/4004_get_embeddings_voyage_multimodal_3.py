### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import json
import os
import requests

import pandas as pd
import voyageai

from dotenv import load_dotenv
from io import BytesIO
from parallel_pandas import ParallelPandas
from pathlib import Path
from PIL import Image
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
parser = argparse.ArgumentParser(description="Get embeddings of meme")
parser.add_argument(
    "--inputfile",
    required=True,
    help="Absolute file path of the .csv file containing meme URLs, OCR responses and meme texts, recognized celebrities and their information for the combined dataset"
)
parser.add_argument(
    "--outputfile",
    required=True,
    help="Path to the output .csv file containing embeddings of meme"
)
parser.add_argument(
    "--inputcred",
    required=True,
    help="Absolute file path of the .env file containing VOYAGE_API_KEY"
)
args = parser.parse_args()

### --------------------------------------------------------------------------------------------------------- ###
### --------------------------Set VOYAGE_API_KEY environment variable for authentication--------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
load_dotenv(Path(args.inputcred))

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------Initialize Voyage AI client-------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

### --------------------------------------------------------------------------------------------------------- ###
### ------------------Create local output directory to store dataframe with meme embeddings------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### ------------------Define a function to call the Voyage AI API to get multimodal embeddings--------------- ###
### --------------------------------------------------------------------------------------------------------- ###
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(100)
)
def get_embeddings(
        text,
        image_url
):
    """
    This function calls the Voyage AI API to get embeddings.
    This leverages code from https://docs.voyageai.com/docs/introduction

            Parameters:
                    text (str): Text on meme.
                    image_url (str): URL to meme.

            Returns:
                    tuple containing

                            - embeddings_response_json (str): Response from calling the Voyage AI API, in JSON string format.
                            - embeddings (list): Embeddings vector from calling the Voyage AI API.
    """
    response = requests.get(image_url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    inputs = [
        [text, img]
    ]
    embeddings_response = vo.multimodal_embed(
        inputs,
        model="voyage-multimodal-3",
        input_type=None
    )
    embeddings = embeddings_response.embeddings[0]
    embeddings_response_json = json.dumps(embeddings_response, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else (o.__dict__ if hasattr(o, "__dict__") else str(o)))
    return embeddings_response_json, embeddings


### --------------------------------------------------------------------------------------------------------- ###
### ------Define main function that applies the get_embeddings function on the meme URLs and meme texts------ ###
### --------------------------------------------------------------------------------------------------------- ###
def main():
    memes = pd.read_csv(
        filepath_or_buffer=args.inputfile,
        encoding="utf-8"
    )
    memes.text = memes.text.fillna("").astype(str).str.strip()
    memes[["embeddings_response_json", "embeddings"]] = memes.p_apply(
        lambda row: pd.Series(
            data=get_embeddings(
                text=row.text,
                image_url=row.meme_s3
            )
        ),
        axis=1
    )
    memes.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfile)), os.path.basename(args.outputfile)),
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
