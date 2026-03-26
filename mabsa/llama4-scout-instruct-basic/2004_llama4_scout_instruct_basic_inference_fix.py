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

ParallelPandas.initialize(n_cpu=32)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------Specify required command line arguments-------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description="Perform MABSA (patch inference results which are originally missing 'The meme views')")
parser.add_argument(
    "--inputfile",
    required=True,
    help="Absolute file path of the .csv file containing MABSA results"
)
parser.add_argument(
    "--outputfiletemp",
    required=True,
    help="Path to the output .csv file containing only the rows where inference was re-attempted (originally missing 'The meme views')"
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
### ------------------Create local output directory to store patched MABSA inference results----------------- ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### ---------------Define a function to call the FIREWORKS AI API to perform MABSA inference----------------- ###
### --------------------------------------------------------------------------------------------------------- ###
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

                            - llama_response_json (str): Response from calling the Fireworks AI API, in JSON string format.
                            - llama_response_text (str): Text content of the response.
                            - retries (int): Number of retry attempts made due to "Invalid Response" ("The meme views" is not found in the inference result) errors.
    """
    retries = 0
    while retries < 10:
        try:
            r = requests.get(image_url)
            r.raise_for_status()
            content = r.content
            base64_bytes = base64.b64encode(content).decode("utf-8")
            llama_response = fireworks.client.ChatCompletion.create(
                model="accounts/fireworks/models/llama4-scout-instruct-basic",
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
                temperature=1,
                top_k=40
            )
            llama_response_json = llama_response.model_dump_json()
            llama_response_text = llama_response.choices[0].message.content
            if "The meme views" not in llama_response_text:
                raise Exception("Invalid Response")
            return llama_response_json, llama_response_text, retries
        except Exception as e:
            if str(e) == "Invalid Response":
                retries += 1
                continue
    return llama_response_json, llama_response_text, retries


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
    mask = memes.llama4_scout_instruct_basic_response_text.apply(lambda x: x.count("The meme views") == 0)
    temp1_df = memes.loc[mask].copy(deep=True)
    temp1_df[["llama4_scout_instruct_basic_response_json", "llama4_scout_instruct_basic_response_text", "retries"]] = temp1_df.p_apply(
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
    memes.loc[temp1_df.index, "llama4_scout_instruct_basic_response_json"] = temp1_df["llama4_scout_instruct_basic_response_json"]
    memes.loc[temp1_df.index, "llama4_scout_instruct_basic_response_text"] = temp1_df["llama4_scout_instruct_basic_response_text"]
    memes.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfile)), os.path.basename(args.outputfile)),
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
