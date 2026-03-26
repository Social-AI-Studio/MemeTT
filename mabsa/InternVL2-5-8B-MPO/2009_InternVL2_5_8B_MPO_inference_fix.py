### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import ast
import json
import os

import pandas as pd

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

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
    do_sample=True,
    max_new_tokens=8192,
    random_seed=42,
    temperature=1,
    top_k=40
)

### --------------------------------------------------------------------------------------------------------- ###
### ------------------Create local output directory to store patched MABSA inference results----------------- ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### ---------------Define a function to call the LMDeploy pipeline to perform MABSA inference---------------- ###
### --------------------------------------------------------------------------------------------------------- ###
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
                            - retries (int): Number of retry attempts made due to "Invalid Response" ("The meme views" is not found in the inference result) errors.
    """
    retries = 0
    while retries < 10:
        try:
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
            if "The meme views" not in internvl_response_text:
                raise Exception("Invalid Response")
            return internvl_response_json, internvl_response_text, retries
        except Exception as e:
            if str(e) == "Invalid Response":
                retries += 1
                continue
    return internvl_response_json, internvl_response_text, retries


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
    mask = memes.InternVL2_5_8B_MPO_response_text.apply(lambda x: x.count("The meme views") == 0)
    temp1_df = memes.loc[mask].copy(deep=True)
    temp1_df[["InternVL2_5_8B_MPO_response_json", "InternVL2_5_8B_MPO_response_text", "retries"]] = temp1_df.apply(
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
    memes.loc[temp1_df.index, "InternVL2_5_8B_MPO_response_json"] = temp1_df["InternVL2_5_8B_MPO_response_json"]
    memes.loc[temp1_df.index, "InternVL2_5_8B_MPO_response_text"] = temp1_df["InternVL2_5_8B_MPO_response_text"]
    memes.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfile)), os.path.basename(args.outputfile)),
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
