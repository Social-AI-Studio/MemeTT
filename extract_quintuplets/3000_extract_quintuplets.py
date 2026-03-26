### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import json
import os
import re
import string

import fireworks.client
import pandas as pd

from dotenv import load_dotenv
from parallel_pandas import ParallelPandas
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

ParallelPandas.initialize(n_cpu=32)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------Specify required command line arguments-------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description="Extract quintuplets")
parser.add_argument(
    "--inputfile",
    required=True,
    help="Absolute file path of the .csv file containing MABSA results"
)
parser.add_argument(
    "--outputfile",
    required=True,
    help="Path to the output .csv file containing extracted quintuplets"
)
parser.add_argument(
    "--textcolname",
    required=True,
    help="Name of the Pandas column where MABSA results are stored"
)
parser.add_argument(
    "--inputprompts",
    required=True,
    help="Absolute file path of the JSON file containing prompts and prompt templates used for inference"
)
parser.add_argument(
    "--inputcred",
    required=True,
    help="Absolute file path of the .env file containing Fireworks API key"
)
args = parser.parse_args()

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------Set FIREWORKS_API_KEY environment variable for authentication-------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
load_dotenv(Path(args.inputcred))

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------Initialize Fireworks API client----------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
fireworks.client.api_key = os.getenv("FIREWORKS_API_KEY")

### --------------------------------------------------------------------------------------------------------- ###
### ---------------Create local output directory to store dataframe containing extracted quintuplets--------- ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### -------------------------Define a function to use regex to extract quintuplets--------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
def extract_quintuplets_initial(
        text
):
    """
    This function takes in a text and returns a list of tuples (viewpoint, target, aspect, opinion, sentiment).

            Parameters:
                    text (str): Input text

            Returns:
                    results (list): List of tuples (viewpoint, target, aspect, opinion, sentiment).
    """
    text = re.sub(r"</?(?:b|i|u|ins)>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*|\{|\}", "", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)
    pattern = r"(The meme views ((?:(?! with a \S+ sentiment because ).)+) with a (.+?) sentiment because (?:(?:its|his|her|their)(?:/(?:its|his|her|their))*) (.+?) (?:(?:is|are|was|were)(?:/(?:is|are|was|were))*) seen as (.+?)[.!?])"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    results = []
    for match in matches:
        viewpoint, target, sentiment, aspect, opinion = match
        sentiment = sentiment.strip().lower().translate(str.maketrans("", "", string.punctuation))
        if viewpoint.strip() and target.strip() and target.strip().lower().translate(str.maketrans("", "", string.punctuation)) not in ("the target", "the targets", "target", "targets") \
                and aspect.strip() and aspect.strip().lower().translate(str.maketrans("", "", string.punctuation)) not in ("the aspect", "the aspects", "aspect", "aspects", "they", "it") \
                and opinion.strip() and opinion.strip().lower().translate(str.maketrans("", "", string.punctuation)) not in ("the opinion", "the opinions", "opinion", "opinions") \
                and sentiment in ("positive", "neutral", "negative"):
            results.append((viewpoint.strip(), target.strip(), aspect.strip(), opinion.strip(), sentiment))
    return results


### --------------------------------------------------------------------------------------------------------- ###
### ----------------------Define another function to use regex to extract quintuplets------------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
def extract_quintuplets_final(
        text
):
    """
    This function takes in a text and returns a list of tuples (viewpoint, target, aspect, opinion, sentiment).

            Parameters:
                    text (str): Input text

            Returns:
                    results (list): List of tuples (viewpoint, target, aspect, opinion, sentiment).
    """
    text = re.sub(r"</?(?:b|i|u|ins)>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*|\{|\}", "", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)
    pattern = r"(The meme views ((?:(?! with a \S+ sentiment because ).)+) with a (.+?) sentiment because (?:(?:its|his|her|their)(?:/(?:its|his|her|their))*\s+)?(.+?)\s+(?:(?:is|are|was|were)(?:/(?:is|are|was|were))*) seen as (.+?)[.!?])"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    results = []
    for match in matches:
        viewpoint, target, sentiment, aspect, opinion = match
        sentiment = sentiment.strip().lower().translate(str.maketrans("", "", string.punctuation))
        if viewpoint.strip() and target.strip() and aspect.strip() and opinion.strip() and sentiment in ("positive", "neutral", "negative"):
            results.append((viewpoint.strip(), target.strip(), aspect.strip(), opinion.strip(), sentiment))
    return results


### --------------------------------------------------------------------------------------------------------- ###
### ------------------Define a function to call the FIREWORKS AI API to extract quintuplets------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(max_attempt_number=100)
)
def inference(
        system_prompt,
        user_prompt
):
    """
    This function calls the Fireworks AI API.
    This leverages code from https://docs.fireworks.ai/guides/querying-text-models

            Parameters:
                    system_prompt (str): System prompt passed to Fireworks AI API.
                    user_prompt (str): User prompt passed to Fireworks AI API.

            Returns:
                    tuple containing

                            - deepseek_response_json (str): Response from calling the Fireworks AI API, in JSON string format.
                            - deepseek_response_text (str): Text content of the response.
    """
    deepseek_response = fireworks.client.ChatCompletion.create(
        model="accounts/fireworks/models/deepseek-r1-basic",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":  user_prompt}
        ],
        max_tokens=32768,
        reasoning_effort="high",
        response_format={"type": "json_object"},
        temperature=0,
        top_k=1
    )
    deepseek_response_json = deepseek_response.model_dump_json()
    deepseek_response_text = deepseek_response.choices[0].message.content
    return deepseek_response_json, deepseek_response_text


### --------------------------------------------------------------------------------------------------------- ###
### ------------Define main function that applies the extract_quintuplets and inference functions------------ ###
### --------------------------------------------------------------------------------------------------------- ###
def main():
    memes = pd.read_csv(
        filepath_or_buffer=args.inputfile,
        encoding="utf-8"
    )
    with open(args.inputprompts, mode="r", encoding="utf-8") as file:
        prompts = json.load(file)
    memes["extracted_quintuplets_regex"] = memes[args.textcolname].p_apply(
        lambda x: extract_quintuplets_initial(
            text=x
        )
    )
    memes[["deepseek_r1_basic_response_json", "deepseek_r1_basic_response_text"]] = memes.p_apply(
        lambda row: inference(
            system_prompt=prompts["system_prompt"],
            user_prompt=prompts["user_prompt"].format(analysis=row[args.textcolname])
        ) if len(row["extracted_quintuplets_regex"]) != 1 or row[args.textcolname].count("The meme views") != 1 else (None, None),
        axis=1,
        result_type="expand"
    )
    memes["deepseek_r1_basic_response_text_truncated"] = memes.deepseek_r1_basic_response_text.p_apply(
        lambda x: re.sub(r"<think>.*?</think>", "", x, flags=re.DOTALL) if isinstance(x, str) and "<think>" in x and "</think>" in x else x
    )
    memes.deepseek_r1_basic_response_text_truncated = memes.deepseek_r1_basic_response_text_truncated.p_apply(lambda x: re.search(r'({.*?})', x, re.DOTALL).group(1) if isinstance(x, str) and re.search(r'({.*?})', x, re.DOTALL) else x)
    memes["extracted_quintuplets_deepseek"] = memes.deepseek_r1_basic_response_text_truncated.p_apply(
        lambda x: extract_quintuplets_final(
            text=x
        ) if not pd.isna(x) and x is not None else None
    )
    memes["extracted_quintuplets_final"] = memes.p_apply(
        lambda row: row["extracted_quintuplets_deepseek"] if row["extracted_quintuplets_deepseek"] not in [None, []] else row["extracted_quintuplets_regex"],
        axis=1
    )
    memes_exploded = memes.explode(
        "extracted_quintuplets_final",
        ignore_index=True
    )
    memes_exploded.extracted_quintuplets_final = memes_exploded.extracted_quintuplets_final.p_apply(lambda x: x if isinstance(x, tuple) else (None, None, None, None, None))
    memes_exploded[["viewpoint", "target", "aspect", "opinion", "sentiment"]] = pd.DataFrame(
        memes_exploded["extracted_quintuplets_final"].tolist(),
        index=memes_exploded.index
    )
    memes_exploded.target = memes_exploded.target.p_apply(lambda x: re.sub(r'^[\'"](?:the\s+)?target of\s*(.*)[\'"]$', r'\1', re.sub(r"^(?:the\s+)?target of\s*", "", x, flags=re.IGNORECASE), flags=re.IGNORECASE) if isinstance(x, str) else x)
    memes_exploded.aspect = memes_exploded.aspect.p_apply(lambda x: re.sub(r'^[\'"](?:the\s+)?aspect of\s*(.*)[\'"]$', r'\1', re.sub(r"^(?:the\s+)?aspect of\s*", "", x, flags=re.IGNORECASE), flags=re.IGNORECASE) if isinstance(x, str) else x)
    memes_exploded.aspect = memes_exploded.aspect.p_apply(lambda x: re.search(r'^\s*[\'"]?\s*aspects?\s+\((.*?)\)[\'"]?\s*$', x, re.IGNORECASE).group(1) if isinstance(x, str) and re.search(r'^\s*[\'"]?\s*aspects?\s+\((.*?)\)[\'"]?\s*$', x, re.IGNORECASE) else x)
    memes_exploded.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfile)), os.path.basename(args.outputfile)),
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
