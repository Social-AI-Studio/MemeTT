### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import json
import time

from lightning_sdk import Machine, Studio
from time import perf_counter

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------------Run studios---------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
llm_judge_evaluate_homogeneity_mixed_embed_v4_0_FINCH_start = perf_counter()
llm_judge_evaluate_homogeneity_mixed_embed_v4_0_FINCH_studio = Studio("6001_llm_judge_evaluate_homogeneity")
llm_judge_evaluate_homogeneity_mixed_embed_v4_0_FINCH_studio.start(Machine.DATA_PREP)
time.sleep(100)
llm_judge_evaluate_homogeneity_mixed_embed_v4_0_FINCH_studio.run("python -m pip install --upgrade pip")
llm_judge_evaluate_homogeneity_mixed_embed_v4_0_FINCH_studio.run("pip install -r /teamspace/uploads/requirements_OpenAI.txt")
llm_judge_evaluate_homogeneity_mixed_embed_v4_0_FINCH_studio.run("python 6001_llm_judge_evaluate_homogeneity.py \
    --inputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001O_completed.csv \
    --clustercol cluster \
    --outputfile /teamspace/studios/6001-llm-judge-evaluate-homogeneity/evaluations/memes_stage6001O1_completed.csv \
    --inputprompts /teamspace/uploads/prompts_homogeneity_evaluation.json \
    --inputcred /teamspace/uploads/OPENAI_CREDENTIALS.env")
llm_judge_evaluate_homogeneity_mixed_embed_v4_0_FINCH_studio.stop()
llm_judge_evaluate_homogeneity_mixed_embed_v4_0_FINCH_stop = perf_counter()
llm_judge_evaluate_homogeneity_mixed_embed_v4_0_FINCH_time = llm_judge_evaluate_homogeneity_mixed_embed_v4_0_FINCH_stop - llm_judge_evaluate_homogeneity_mixed_embed_v4_0_FINCH_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "llm_judge_evaluate_homogeneity_mixed_embed_v4_0_FINCH_time": str(llm_judge_evaluate_homogeneity_mixed_embed_v4_0_FINCH_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="llm_judge_evaluate_homogeneity_mixed_embed_v4_0_FINCH_time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)