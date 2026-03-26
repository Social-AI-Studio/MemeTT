### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import ast
import os
import json
import requests

import pandas as pd

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmCategory,
    HarmBlockThreshold,
    HttpOptions,
    SafetySetting,
    Part
)
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
    help="Absolute file path of the Google Cloud service account JSON key file"
)
parser.add_argument(
    "--projectdetails",
    required=True,
    help="Absolute file path of the .env file containing GOOGLE CLOUD PROJECT and GOOGLE_CLOUD_LOCATION and GOOGLE_GENAI_USE_VERTEXAI"
)
args = parser.parse_args()

### --------------------------------------------------------------------------------------------------------- ###
### ----------------Set GOOGLE_APPLICATION_CREDENTIALS environment variable for authentication--------------- ###
### --------------------------------------------------------------------------------------------------------- ###
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.inputcred
load_dotenv(Path(args.projectdetails))

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------Initialize Gen AI client-------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
client = genai.Client(http_options=HttpOptions(api_version="v1"))

### --------------------------------------------------------------------------------------------------------- ###
### ---------------Create local output directory to store dataframe containing MABSA inferences-------------- ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### ----------------Define a function to call the Vertex AI API to perform MABSA inference------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(100)
)
def inference(
        image_url,
        system_prompt,
        user_prompt
):
    """
    This function calls the Vertex AI API.
    This leverages code from https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding

            Parameters:
                    image_url (str): URL to meme.
                    system_prompt (str): System prompt passed to Vertex AI API.
                    user_prompt (str): User prompt passed to Vertex AI API.

            Returns:
                    tuple containing

                            - gemini_response_json (str): Response from calling the Vertex AI API, in JSON string format.
                            - gemini_response_text (str): Text content of the response.
    """
    r = requests.get(image_url)
    r.raise_for_status()
    content = r.content
    contents = [
        Part.from_bytes(
            data=content,
            mime_type="image/png"
        ),
        user_prompt
    ]
    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=contents,
        config=GenerateContentConfig(
            max_output_tokens=8192,
            system_instruction=system_prompt,
            safety_settings=[
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.OFF,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.OFF,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.OFF,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.OFF,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                    threshold=HarmBlockThreshold.OFF,
                ),
            ],
            seed=42,
            temperature=0,
            top_k=1
        )
    )
    # JSON serializer. via
    # https://speckle.community/t/ids-in-speckle-server-not-matching-those-retrieved-from-api/5880/7
    gemini_response_json = json.dumps(gemini_response, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else (o.__dict__ if hasattr(o, "__dict__") else str(o)))
    gemini_response_text = gemini_response.text
    return gemini_response_json, gemini_response_text


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
    memes[["gemini_2_0_flash_001_response_json", "gemini_2_0_flash_001_response_text"]] = memes.p_apply(
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
