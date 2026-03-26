### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import os
import requests

import pandas as pd

from google.cloud import vision_v1
from google.cloud.vision_v1 import AnnotateImageResponse
from parallel_pandas import ParallelPandas
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

ParallelPandas.initialize(n_cpu=32)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------Specify required command line arguments-------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description="Detect text in memes")
parser.add_argument(
    "--inputfile",
    required=True,
    help="Absolute file path of .csv file containing file paths and URLs of TotalDefMeme, Harm-P and Harm-C memes"
)
parser.add_argument(
    "--outputfile",
    required=True,
    help="Path to the output .csv file containing meme URLs, OCR responses and meme texts for the combined dataset"
)
parser.add_argument(
    "--inputcred",
    required=True,
    help="Absolute file path of the Google Cloud service account JSON key file"
)
args = parser.parse_args()

### --------------------------------------------------------------------------------------------------------- ###
### ----------------Set GOOGLE_APPLICATION_CREDENTIALS environment variable for authentication--------------- ###
### --------------------------------------------------------------------------------------------------------- ###
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.inputcred

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------Initialize Google Cloud Vision API client------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
client = vision_v1.ImageAnnotatorClient()

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------Create a directory to store .csv file containing meme text---------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### ---------------Define a function to call the Google Cloud Vision API to detect text on meme-------------- ###
### --------------------------------------------------------------------------------------------------------- ###
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(50)
)
def detect_text(
        image_url
):
    """
    This function calls the Google Cloud Vision API to perform OCR and locate the text on the meme.
    This leverages code from https://cloud.google.com/vision/docs/ocr#detect_text_in_a_local_image

            Parameters:
                    image_url (str): URL of meme.

            Returns:
                    tuple containing

                            - response_json (str): Response from calling the Google Cloud Vision API, in JSON string format.
                            - ocr_text (str | None): Text on the meme if text is extracted by Google Cloud Vision API else None.
    """
    r = requests.get(image_url)
    r.raise_for_status()
    content = r.content
    image = vision_v1.Image(content=content)
    response = client.text_detection(image=image)
    response_json = AnnotateImageResponse.to_json(response)
    if response.text_annotations:
        ocr_text = response.text_annotations[0].description
        return response_json, ocr_text
    else:
        return response_json, None


### --------------------------------------------------------------------------------------------------------- ###
### ----------Define main function that applies the detect_text function on the memes in the dataset--------- ###
### --------------------------------------------------------------------------------------------------------- ###
def main():
    memes = pd.read_csv(
        filepath_or_buffer=args.inputfile,
        encoding="utf-8"
    )
    memes[["response_detect_text", "text"]] = memes.meme_s3.p_apply(
        lambda x: pd.Series(
            data=detect_text(
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
