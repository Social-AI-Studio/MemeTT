### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import base64
import os
import requests

import pandas as pd

from dotenv import load_dotenv
import cohere
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
parser = argparse.ArgumentParser(description="Get embeddings of memes")
parser.add_argument(
    "--inputfile",
    required=True,
    help="Absolute file path of the .csv file containing meme URLs, OCR responses and meme texts, recognized celebrities and their information for the combined dataset"
)
parser.add_argument(
    "--outputfile",
    required=True,
    help="Path to the output .csv file containing embeddings of memes"
)
parser.add_argument(
    "--inputcred",
    required=True,
    help="Absolute file path of the .env file containing COHERE_API_KEY"
)
args = parser.parse_args()

### --------------------------------------------------------------------------------------------------------- ###
### --------------------------Set COHERE_API_KEY environment variable for authentication--------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
load_dotenv(Path(args.inputcred))

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------Initialize Cohere AI client-------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

### --------------------------------------------------------------------------------------------------------- ###
### ------------------Create local output directory to store dataframe with meme embeddings------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### ---------------------Define a function to call the Cohere Embed API to get embeddings-------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(100)
)
def get_embeddings(
        image_url
):
    """
    This function calls the Cohere Embed API to get embeddings.
    This leverages code from https://docs.cohere.com/reference/embed

            Parameters:
                    image_url (str): URL to meme.

            Returns:
                    tuple containing

                            - embeddings_response_json (str): Response from calling the Cohere Embed API, in JSON string format.
                            - embeddings (list): Embeddings vector from calling the Cohere Embed API.
    """
    image = requests.get(image_url)
    stringified_buffer = base64.b64encode(image.content).decode("utf-8")
    image_base64 = f"data:image/png;base64,{stringified_buffer}"
    embeddings_response = co.embed(
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
        images=[image_base64],
    )
    embeddings = embeddings_response.embeddings.float_[0]
    embeddings_response_json = embeddings_response.model_dump_json()
    return embeddings_response_json, embeddings


### --------------------------------------------------------------------------------------------------------- ###
### -------------Define main function that applies the get_embeddings function on the meme URLs-------------- ###
### --------------------------------------------------------------------------------------------------------- ###
def main():
    memes = pd.read_csv(
        filepath_or_buffer=args.inputfile,
        encoding="utf-8"
    )
    memes[["embeddings_response_json", "embeddings"]] = memes.meme_s3.p_apply(
        lambda x: pd.Series(
            data=get_embeddings(
                image_url=x
            )
        )
    )
    memes.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfile)), os.path.basename(args.outputfile)),
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
