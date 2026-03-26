### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import ast
import json
import os

import pandas as pd

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
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
args = parser.parse_args()

### --------------------------------------------------------------------------------------------------------- ###
### -----------------------------------------Initialize LMDeploy pipeline------------------------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
pipe = pipeline(
    "OpenGVLab/InternVL2_5-8B-MPO",
    backend_config=TurbomindEngineConfig(
        dtype="bfloat16",
        quant_policy=0,
        revision="4e8ad56dfb972d22caa1bccb60d2c10b41253c89",
        rope_scaling_factor=4,
        session_len=32768
    )
)
gen_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=8192
)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------Create local output directory to store dataframe containing MABSA inferences------------- ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### ---------------Define a function to call the LMDeploy pipeline to perform MABSA inference---------------- ###
### --------------------------------------------------------------------------------------------------------- ###
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(max_attempt_number=10)
)
def inference(
        image_url,
        system_prompt,
        user_prompt
):
    """
    This function calls the LMDeploy pipeline.
    This leverages code from https://lmdeploy.readthedocs.io/en/latest/multi_modal/internvl.html

            Parameters:
                    image_url (str): URL to meme.
                    system_prompt (str): System prompt passed to the LMDeploy pipeline.
                    user_prompt (str): User prompt passed to LMDeploy pipeline.

            Returns:
                    tuple containing

                            - internvl_response_json (str): Response from calling the LMDeploy pipeline, in JSON string format.
                            - internvl_response_text (str): Text content of the response.
    """
    prompts = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]
    internvl_response = pipe(
        prompts,
        gen_config=gen_config
    )
    internvl_response_json = json.dumps(internvl_response, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else (o.__dict__ if hasattr(o, "__dict__") else str(o)))
    internvl_response_text = internvl_response.text
    return internvl_response_json, internvl_response_text


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
    memes[["InternVL2_5_8B_MPO_response_json", "InternVL2_5_8B_MPO_response_text"]] = memes.apply(
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
