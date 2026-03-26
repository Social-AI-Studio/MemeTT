### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import ast
import json
import os

import pandas as pd
import torch

from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForImageTextToText
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
### -----------------------------------------Download model snapshot----------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
snapshot_download(
    "OpenGVLab/InternVL3-8B-hf",
    revision="398b7352534c4b0f922bf4e0d153c358f07b9c34",
    local_dir="model",
    ignore_patterns=["*.pt", "*.bin"],
)

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------------Load model----------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
model = AutoModelForImageTextToText.from_pretrained(
    "model",
    torch_dtype=torch.bfloat16
).eval().cuda()
processor = AutoProcessor.from_pretrained(
    "model"
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
### ------------Define a function to call the InternVL3-8B-hf model to perform MABSA inference--------------- ###
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
    This function calls the OpenGVLab/InternVL3-8B-hf.
    This leverages code from https://huggingface.co/OpenGVLab/InternVL3-8B-hf

            Parameters:
                    image_url (str): URL to meme.
                    system_prompt (str): System prompt passed to the OpenGVLab/InternVL3-8B-hf model.
                    user_prompt (str): User prompt passed to the OpenGVLab/InternVL3-8B-hf model.

            Returns:
                    tuple containing

                            - prompt_token_count (int): Number of prompt tokens.
                            - total_token_count (int): Total number of tokens (prompt + response).
                            - internvl_response_text (str): Text content of the response.
    """
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(
        model.device,
        dtype=torch.bfloat16
    )
    prompt_token_count = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=8192,
            num_beams=1,
            use_cache=True
        )
    total_token_count = generate_ids.shape[1]
    internvl_response_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return prompt_token_count, total_token_count, internvl_response_text


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
    memes[["prompt_token_count", "total_token_count", "InternVL3_8B_hf_response_text"]] = memes.apply(
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
