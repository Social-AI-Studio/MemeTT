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

ParallelPandas.initialize(
    n_cpu=4,
    split_factor=2
)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------Specify required command line arguments-------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description="Perform MABSA (patch inference results which are originally blank/non-string or missing 'The meme views')")
parser.add_argument(
    "--inputfile",
    required=True,
    help="Absolute file path of the .csv file containing MABSA results"
)
parser.add_argument(
    "--outputfiletemp",
    required=True,
    help="Path to the output .csv file containing only the rows where inference was re-attempted (originally blank/non-string or missing 'The meme views')"
)
parser.add_argument(
    "--outputfile",
    required=True,
    help="Path to the output .csv file containing MABSA results patched"
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
### ------------------Create local output directory to store patched MABSA inference results----------------- ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### ----------------Define a function to call the Vertex AI API to perform MABSA inference------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
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
                            - retries (int): Number of retry attempts made due to "Invalid Response" ("The meme views" is not found in the inference result) errors.
    """
    retries = 0
    while retries < 10:
        try:
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
                model="gemini-2.5-pro-exp-03-25",
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
                    temperature=1,
                    top_k=40
                )
            )
            # JSON serializer. via
            # https://speckle.community/t/ids-in-speckle-server-not-matching-those-retrieved-from-api/5880/7
            gemini_response_json = json.dumps(gemini_response, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else (o.__dict__ if hasattr(o, "__dict__") else str(o)))
            gemini_response_text = gemini_response.text
            if "The meme views" not in gemini_response_text:
                raise Exception("Invalid Response")
            return gemini_response_json, gemini_response_text, retries
        except Exception as e:
            if str(e) == "Invalid Response":
                retries += 1
                continue
    return gemini_response_json, gemini_response_text, retries


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
    mask = memes.gemini_2_5_pro_exp_03_25_response_text.apply(lambda x: True if (not isinstance(x, str) or not x.strip() or x.count("The meme views") == 0) else False)
    temp1_df = memes.loc[mask].copy(deep=True)
    temp1_df[["gemini_2_5_pro_exp_03_25_response_json", "gemini_2_5_pro_exp_03_25_response_text", "retries"]] = temp1_df.p_apply(
        lambda row: inference(
            image_url=row["meme_bboxes_drawn_s3"],
            system_prompt=prompts["system_prompt"],
            user_prompt=prompts["user_prompt"] if not (isinstance(row["celebrities"], list) and row["celebrities"]) else prompts["prefix"].format(information=" ".join([prompts["identities"].format(color=celebrity[1], name=celebrity[0], description=celebrity[2]) for celebrity in row["celebrities"]])) + prompts["user_prompt"]
        ),
        axis=1,
        result_type="expand"
    )
    temp1_df.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfiletemp)), os.path.basename(args.outputfiletemp)),
        index=False,
        encoding="utf-8"
    )
    memes.loc[temp1_df.index, "gemini_2_5_pro_exp_03_25_response_json"] = temp1_df["gemini_2_5_pro_exp_03_25_response_json"]
    memes.loc[temp1_df.index, "gemini_2_5_pro_exp_03_25_response_text"] = temp1_df["gemini_2_5_pro_exp_03_25_response_text"]
    memes.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfile)), os.path.basename(args.outputfile)),
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
