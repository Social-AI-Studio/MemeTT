### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import ast
import json
import os

import pandas as pd

from qwen_vl_utils import process_vision_info
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_distributed_environment

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
### ---------------------------------Initialize vllm offline inference pipeline------------------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
llm = LLM(
    dtype="bfloat16",
    gpu_memory_utilization=.9,
    max_seq_len_to_capture=128000,
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    limit_mm_per_prompt={"image": 1},
    revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
    seed=42,
    task="generate"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
sampling_params = SamplingParams(
    temperature=0,
    top_k=1,
    max_tokens=8192,
    stop_token_ids=[],
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
### -------Define a function to call the vllm offline inference pipeline to perform MABSA inference---------- ###
### --------------------------------------------------------------------------------------------------------- ###
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(10)
)
def inference(
        image_url,
        system_prompt,
        user_prompt
):
    """
    This function calls the vllm offline inference pipeline.
    This leverages code from https://github.com/QwenLM/Qwen2.5-VL

            Parameters:
                    image_url (str): URL to meme.
                    system_prompt (str): System prompt passed to vllm offline inference pipeline.
                    user_prompt (str): User prompt passed to vllm offline inference pipeline.

            Returns:
                    tuple containing

                            - qwen_response_json (str): Response from calling the vllm offline inference pipeline, in JSON string format.
                            - qwen_response_text (str): Text content of the response.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url,
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(
        messages,
        return_video_kwargs=False
    )
    mm_data = {}
    mm_data["image"] = image_inputs
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data
    }
    qwen_response = llm.generate(
        [llm_inputs],
        sampling_params=sampling_params
    )
    qwen_response_json= json.dumps(qwen_response, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else (o.__dict__ if hasattr(o, "__dict__") else str(o)))
    qwen_response_text = qwen_response[0].outputs[0].text
    return qwen_response_json, qwen_response_text

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
    memes[["Qwen2_5_VL_7B_Instruct_response_json", "Qwen2_5_VL_7B_Instruct_response_text"]] = memes.apply(
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
    destroy_distributed_environment()

if __name__ == "__main__":
    main()

