### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import ast
import json
import os
import re
import string

import fireworks.client
import json_repair
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from finch import FINCH
from parallel_pandas import ParallelPandas
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

ParallelPandas.initialize(n_cpu=64)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------Specify required command line arguments-------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description="Cluster and generate labels")
parser.add_argument(
    "--inputfile",
    required=True,
    help="Absolute file path of .csv file containing embeddings of viewpoints, targets, aspects, opinions"
)
parser.add_argument(
    "--outputfile",
    required=True,
    help="Path to the output .csv file containing clusters and generated labels"
)
parser.add_argument(
    "--outputfilesubset",
    required=True,
    help="Path to the output .csv file containing clusters and generated labels (subset of columns)"
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
### ------------------------------Load Fireworks credentials from the provided .env file--------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
load_dotenv(Path(args.inputcred))

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------------Set Fireworks API key for the SDK----------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
fireworks.client.api_key = os.getenv("FIREWORKS_API_KEY")

### --------------------------------------------------------------------------------------------------------- ###
### --------Create local output directory to store dataframe containing clusters and generated labels-------- ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### -----------------------------------Define a function to cluster embeddings------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
def cluster_embeddings(embeddings):
    """
    This function calls the FINCH algorithm to cluster embeddings and retrieve numeric cluster labels.
    This leverages code from https://github.com/ssarfraz/FINCH-Clustering

            Parameters:
                    embeddings (numpy.ndarray): Input matrix with observations as rows and features as columns.

            Returns:
                    cluster_label_numeric (numpy.ndarray): Array of numeric cluster labels.
    """
    c, num_clust, req_c = FINCH(
        data=embeddings,
        distance="cosine",
        verbose=False
    )
    partition = 0
    cluster_label_numeric = c[:, partition]
    return cluster_label_numeric


### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------Define a function to call the FIREWORKS AI API---------------------------- ###
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
                            qwen_response_json (str): Response from calling the Fireworks AI API, in JSON string format.
                            qwen_response_text (str): Text content of the response.
                            retries (int): Flag indicating whether the fallback sampling call was used:
                                - 0 = first API call returned finish_reason == "stop"
                                - 1 = fallback second API call was used
    """
    qwen_response = fireworks.client.ChatCompletion.create(
        model="accounts/fireworks/models/qwen3-235b-a22b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":  user_prompt}
        ],
        max_tokens=16384,
        reasoning_effort="high",
        temperature=0,
        top_k=1
    )
    qwen_response_json = qwen_response.model_dump_json()
    qwen_response_text = qwen_response.choices[0].message.content
    if qwen_response.choices[0].finish_reason == "stop":
        return qwen_response_json, qwen_response_text, 0
    qwen_response = fireworks.client.ChatCompletion.create(
        model="accounts/fireworks/models/qwen3-235b-a22b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":  user_prompt}
        ],
        max_tokens=16384,
        reasoning_effort="high",
        temperature=1,
        top_k=40
    )
    qwen_response_json = qwen_response.model_dump_json()
    qwen_response_text = qwen_response.choices[0].message.content
    return qwen_response_json, qwen_response_text, 1


### --------------------------------------------------------------------------------------------------------- ###
### -------------Define main function that applies the cluster_embeddings and inference functions------------ ###
### --------------------------------------------------------------------------------------------------------- ###
def main():
    memes = pd.read_csv(
        filepath_or_buffer=args.inputfile,
        encoding="utf-8"
    )
    with open(args.inputprompts, mode="r", encoding="utf-8") as file:
        prompts = json.load(file)
    memes.target_embeddings = memes.target_embeddings.p_apply(lambda x: ast.literal_eval(x))
    memes.aspect_embeddings = memes.aspect_embeddings.p_apply(lambda x: ast.literal_eval(x))
    memes.opinion_embeddings = memes.opinion_embeddings.p_apply(lambda x: ast.literal_eval(x))

    memes["target_cluster_initial"] = cluster_embeddings(np.vstack(memes["target_embeddings"].values))
    memes = memes.join(
        memes
        .groupby("target_cluster_initial")
        .p_apply(
            lambda df: pd.Series(
                (None, df["target"].iloc[0], None)
                if df["target"].str.lower().nunique() == 1
                else inference(
                    system_prompt=prompts["system_prompt_target_cluster_label"],
                    user_prompt=prompts["user_prompt_target_cluster"].format(targets="; ".join(sorted(df["target"], key=str.lower)))
                ),
                index=["target_qwen3_235b_a22b_response_json", "target_cluster_label", "target_retries"]
            )
        ),
        on="target_cluster_initial"
    )
    memes.target_cluster_label = memes.target_cluster_label.p_apply(
        lambda x: re.sub(r"<think>.*?</think>", "", x, flags=re.DOTALL) if isinstance(x, str) and "<think>" in x and "</think>" in x else x
    )
    memes.target_cluster_label = memes.target_cluster_label.str.strip()
    memes.target_cluster_label = memes.p_apply(
        lambda row: row.target_cluster_label if isinstance(row.target_qwen3_235b_a22b_response_json, str) and json.loads(row.target_qwen3_235b_a22b_response_json).get("choices", [{}])[0].get("finish_reason") == "stop" else row.target,
        axis=1
    )
    memes[["check_target_membership_qwen3_235b_a22b_response_json", "check_target_membership_verdict", "check_target_membership_retries"]] = memes.p_apply(
        lambda row: inference(
            system_prompt=prompts["system_prompt_check"],
            user_prompt=prompts["user_prompt_check"].format(text=row.target, label=row.target_cluster_label)
        ) if str(row.target).strip().lower() != str(row.target_cluster_label).strip().lower() else (None, "no", None),
        axis=1,
        result_type="expand"
    )
    memes.check_target_membership_verdict = memes.check_target_membership_verdict.p_apply(
        lambda x: re.sub(r"<think>.*?</think>", "", x, flags=re.DOTALL) if isinstance(x, str) and "<think>" in x and "</think>" in x else x
    )
    memes.check_target_membership_verdict = memes.check_target_membership_verdict.str.replace(r"\s+|[{}]+".format(re.escape(string.punctuation)), "", regex=True).str.lower()
    memes.target_cluster_label = memes.target_cluster_label.where(
        memes.check_target_membership_verdict == "no",
        memes.target
    )

    memes["aspect_cluster_initial"] = memes.groupby("target_cluster_label")["aspect_embeddings"].transform(
        lambda embeddings: (
            np.zeros(len(embeddings), dtype=int) if np.unique(np.vstack(embeddings), axis=0).shape[0] == 1 else cluster_embeddings(np.vstack(embeddings))
        )
    )
    memes = memes.join(
        memes
        .groupby(["target_cluster_label", "aspect_cluster_initial"])
        .p_apply(
            lambda df: pd.Series(
                (None, df["aspect"].iloc[0], None)
                if df["aspect"].str.lower().nunique() == 1
                else inference(
                    system_prompt=prompts["system_prompt_aspect_cluster_label"].format(target=df["target_cluster_label"].iloc[0]),
                    user_prompt=prompts["user_prompt_aspect_cluster"].format(aspects="; ".join(sorted(df["aspect"], key=str.lower)))
                ),
                index=["aspect_qwen3_235b_a22b_response_json", "aspect_cluster_label", "aspect_retries"]
            )
        ),
        on=["target_cluster_label", "aspect_cluster_initial"]
    )
    memes.aspect_cluster_label = memes.aspect_cluster_label.p_apply(
        lambda x: re.sub(r"<think>.*?</think>", "", x, flags=re.DOTALL) if isinstance(x, str) and "<think>" in x and "</think>" in x else x
    )
    memes.aspect_cluster_label = memes.aspect_cluster_label.str.strip()
    memes.aspect_cluster_label = memes.p_apply(
        lambda row: row.aspect_cluster_label if isinstance(row.aspect_qwen3_235b_a22b_response_json, str) and json.loads(row.aspect_qwen3_235b_a22b_response_json).get("choices", [{}])[0].get("finish_reason") == "stop" else row.aspect,
        axis=1
    )
    memes[["check_aspect_membership_qwen3_235b_a22b_response_json", "check_aspect_membership_verdict", "check_aspect_membership_retries"]] = memes.p_apply(
        lambda row: inference(
            system_prompt=prompts["system_prompt_check"],
            user_prompt=prompts["user_prompt_check"].format(text=row.aspect, label=row.aspect_cluster_label)
        ) if str(row.aspect).strip().lower() != str(row.aspect_cluster_label).strip().lower() else (None, "no", None),
        axis=1,
        result_type="expand"
    )
    memes.check_aspect_membership_verdict = memes.check_aspect_membership_verdict.p_apply(
        lambda x: re.sub(r"<think>.*?</think>", "", x, flags=re.DOTALL) if isinstance(x, str) and "<think>" in x and "</think>" in x else x
    )
    memes.check_aspect_membership_verdict = memes.check_aspect_membership_verdict.str.replace(r"\s+|[{}]+".format(re.escape(string.punctuation)), "", regex=True).str.lower()
    memes.aspect_cluster_label = memes.aspect_cluster_label.where(
        memes.check_aspect_membership_verdict == "no",
        memes.aspect
    )
    memes = memes.join(
        memes
        .groupby("target_cluster_label")
        .p_apply(
            lambda grp: pd.Series(
                (None, {}, None)
                if grp.aspect_cluster_label.nunique() == 1
                else inference(
                    system_prompt=prompts["system_prompt_merge"],
                    user_prompt=prompts["user_prompt_merge"].format(
                        labels="; ".join(sorted(grp.aspect_cluster_label.unique(), key=str.lower))
                    )
                ),
                index=[
                    "merge_aspect_qwen3_235b_response_json",
                    "merge_aspect",
                    "merge_aspect_retries",
                ],
            )
        ),
        on="target_cluster_label"
    )
    memes.merge_aspect = memes.merge_aspect.p_apply(
        lambda x: re.sub(r"<think>.*?</think>", "", x, flags=re.DOTALL) if isinstance(x, str) and "<think>" in x and "</think>" in x else x
    )
    memes.merge_aspect = memes.merge_aspect.p_apply(lambda x: x.strip() if isinstance(x, str) else x)
    memes.merge_aspect = memes.merge_aspect.p_apply(
        lambda x: x if isinstance(x, dict) else json_repair.loads(x) if isinstance(x, str) and x.strip() else {}
    )
    memes["aspect_cluster_label_final"] = memes.p_apply(
        lambda row: (
            row.aspect_cluster_label
            if not isinstance(row.merge_aspect, dict) or not row.merge_aspect
            else next(
                (
                    k
                    for k, v in row.merge_aspect.items()
                    if any(
                        row.aspect_cluster_label.lower().translate(str.maketrans("", "", string.punctuation)) == x.strip().lower().translate(str.maketrans("", "", string.punctuation))
                        for x in v.split(";")
                    )
                ),
                row.aspect_cluster_label
            )
        ),
        axis=1
    )

    memes.sentiment = memes.sentiment.str.strip()
    memes["opinion_cluster_initial"] = memes.groupby(["target_cluster_label", "aspect_cluster_label_final", "sentiment"])["opinion_embeddings"].transform(
        lambda embeddings: (
            np.zeros(len(embeddings), dtype=int) if np.unique(np.vstack(embeddings), axis=0).shape[0] == 1 else cluster_embeddings(np.vstack(embeddings))
        )
    )
    memes = memes.join(
        memes
        .groupby([
            "target_cluster_label",
            "aspect_cluster_label_final",
            "sentiment",
            "opinion_cluster_initial"
        ])
        .p_apply(
            lambda df: pd.Series(
                (None, df["opinion"].iloc[0], None)
                if df["opinion"].str.lower().nunique() == 1
                else inference(
                    system_prompt=prompts["system_prompt_opinion_cluster_label"].format(target=df["target_cluster_label"].iloc[0], aspect=df["aspect_cluster_label_final"].iloc[0], sentiment=df["sentiment"].iloc[0]),
                    user_prompt=prompts["user_prompt_opinion_cluster"].format(opinions="; ".join(sorted(df["opinion"], key=str.lower)))
                ),
                index=["opinion_qwen3_235b_a22b_response_json", "opinion_cluster_label", "opinion_retries"]
            )
        ),
        on=["target_cluster_label", "aspect_cluster_label_final", "sentiment", "opinion_cluster_initial"]
    )
    memes.opinion_cluster_label = memes.opinion_cluster_label.p_apply(
        lambda x: re.sub(r"<think>.*?</think>", "", x, flags=re.DOTALL) if isinstance(x, str) and "<think>" in x and "</think>" in x else x
    )
    memes.opinion_cluster_label = memes.opinion_cluster_label.str.strip()
    memes.opinion_cluster_label = memes.p_apply(
        lambda row: row.opinion_cluster_label if isinstance(row.opinion_qwen3_235b_a22b_response_json, str) and json.loads(row.opinion_qwen3_235b_a22b_response_json).get("choices", [{}])[0].get("finish_reason") == "stop" else row.opinion,
        axis=1
    )
    memes[["check_opinion_membership_qwen3_235b_a22b_response_json", "check_opinion_membership_verdict", "check_opinion_membership_retries"]] = memes.p_apply(
        lambda row: inference(
            system_prompt=prompts["system_prompt_check"],
            user_prompt=prompts["user_prompt_check"].format(text=row.opinion, label=row.opinion_cluster_label)
        ) if str(row.opinion).strip().lower() != str(row.opinion_cluster_label).strip().lower() else (None, "no", None),
        axis=1,
        result_type="expand"
    )
    memes.check_opinion_membership_verdict = memes.check_opinion_membership_verdict.p_apply(
        lambda x: re.sub(r"<think>.*?</think>", "", x, flags=re.DOTALL) if isinstance(x, str) and "<think>" in x and "</think>" in x else x
    )
    memes.check_opinion_membership_verdict = memes.check_opinion_membership_verdict.str.replace(r"\s+|[{}]+".format(re.escape(string.punctuation)), "", regex=True).str.lower()
    memes.opinion_cluster_label = memes.opinion_cluster_label.where(
        memes.check_opinion_membership_verdict == "no",
        memes.opinion
    )
    memes = memes.join(
        memes
        .groupby(
            ["target_cluster_label", "aspect_cluster_label_final", "sentiment"]
        )
        .p_apply(
            lambda grp: pd.Series(
                (None, {}, None)
                if grp.opinion_cluster_label.nunique() == 1
                else inference(
                    system_prompt=prompts["system_prompt_merge"],
                    user_prompt=prompts["user_prompt_merge"].format(
                        labels="; ".join(
                            sorted(grp.opinion_cluster_label.unique(), key=str.lower)
                        )
                    )
                ),
                index=[
                    "merge_opinion_qwen3_235b_response_json",
                    "merge_opinion",
                    "merge_opinion_retries",
                ],
            )
        ),
        on=["target_cluster_label", "aspect_cluster_label_final", "sentiment"],
    )
    memes.merge_opinion = memes.merge_opinion.p_apply(
        lambda x: re.sub(r"<think>.*?</think>", "", x, flags=re.DOTALL) if isinstance(x, str) and "<think>" in x and "</think>" in x else x
    )
    memes.merge_opinion = memes.merge_opinion.p_apply(lambda x: x.strip() if isinstance(x, str) else x)
    memes.merge_opinion = memes.merge_opinion.p_apply(
        lambda x: x if isinstance(x, dict) else json_repair.loads(x) if isinstance(x, str) and x.strip() else {}
    )
    memes["opinion_cluster_label_final"] = memes.p_apply(
        lambda row: (
            row.opinion_cluster_label
            if not isinstance(row.merge_opinion, dict) or not row.merge_opinion
            else next(
                (
                    k
                    for k, v in row.merge_opinion.items()
                    if any(
                        row.opinion_cluster_label.lower().translate(str.maketrans("", "", string.punctuation)) == x.strip().lower().translate(str.maketrans("", "", string.punctuation))
                        for x in v.split(";")
                    )
                ),
                row.opinion_cluster_label
            )
        ),
        axis=1
    )

    memes["narrative"] = memes.p_apply(
        lambda row: f"The meme(s) view {row['target_cluster_label']} with a {row['sentiment']} sentiment because its/his/her/their {row['aspect_cluster_label_final']} is/are seen as {row['opinion_cluster_label_final']}.",
        axis=1
    )
    memes.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfile)), os.path.basename(args.outputfile)),
        index=False,
        encoding="utf-8"
    )
    memes_subset = memes[["meme_s3", "source", "target", "aspect", "opinion", "sentiment", "viewpoint", "target_cluster_label", "aspect_cluster_label_final", "opinion_cluster_label_final", "narrative"]].copy(deep=True)
    memes_subset.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfilesubset)), os.path.basename(args.outputfilesubset)),
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
