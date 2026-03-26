### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import os

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
parser = argparse.ArgumentParser(description="Get text embeddings of meme text")
parser.add_argument(
    "--inputfile",
    required=True,
    help="Absolute file path of the .csv file containing meme URLs, OCR responses and meme texts, recognized celebrities and their information for the combined dataset"
)
parser.add_argument(
    "--outputfileunique",
    required=True,
    help="Path to the output .csv file containing embeddings of unique meme text"
)
parser.add_argument(
    "--outputfile",
    required=True,
    help="Path to the output .csv file containing embeddings of meme text"
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
### ------------------Create local output directory to store dataframe with text embeddings------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### ------------------Define a function to call the Cohere Embed API to get text embeddings------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(100)
)
def get_embeddings(
        text
):
    """
    This function calls the Cohere Embed API to get text embeddings.
    This leverages code from https://docs.cohere.com/reference/embed

            Parameters:
                    text (str): Input text.

            Returns:
                    tuple containing

                            - embeddings_response_json (str): Response from calling the Cohere Embed API, in JSON string format.
                            - embeddings (list): Embeddings vector from calling the Cohere Embed API.
    """
    embeddings_response = co.embed(
        texts=[text],
        model="embed-v4.0",
        input_type="clustering",
        embedding_types=["float"],
    )
    embeddings = embeddings_response.embeddings.float_[0]
    embeddings_response_json = embeddings_response.model_dump_json()
    return embeddings_response_json, embeddings


### --------------------------------------------------------------------------------------------------------- ###
### -------------Define main function that applies the get_embeddings function on the meme texts------------- ###
### --------------------------------------------------------------------------------------------------------- ###
def main():
    memes = pd.read_csv(
        filepath_or_buffer=args.inputfile,
        encoding="utf-8"
    )
    memes.text = memes.text.fillna("").astype(str).str.strip()
    texts = memes.loc[memes["text"] != "", "text"].drop_duplicates().to_frame("text").reset_index(drop=True)
    texts[["embeddings_response_json", "embeddings"]] = texts.text.p_apply(
        lambda x: pd.Series(
            data=get_embeddings(
                text=x
            )
        )
    )
    texts.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfileunique)), os.path.basename(args.outputfileunique)),
        index=False,
        encoding="utf-8"
    )
    embeddings_map = dict(zip(texts["text"], texts["embeddings"]))
    memes["text_embeddings"] = memes.text.map(embeddings_map).p_apply(lambda vector: vector if isinstance(vector, list) else [])
    memes.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfile)), os.path.basename(args.outputfile)),
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
