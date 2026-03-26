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
llm_judge_evaluate_homogeneity_viewpoint_text_embedding_005_KMEANS_start = perf_counter()
llm_judge_evaluate_homogeneity_viewpoint_text_embedding_005_KMEANS_studio = Studio("6001_llm_judge_evaluate_homogeneity")
llm_judge_evaluate_homogeneity_viewpoint_text_embedding_005_KMEANS_studio.start(Machine.DATA_PREP)
time.sleep(100)
llm_judge_evaluate_homogeneity_viewpoint_text_embedding_005_KMEANS_studio.run("python -m pip install --upgrade pip")
llm_judge_evaluate_homogeneity_viewpoint_text_embedding_005_KMEANS_studio.run("pip install -r /teamspace/uploads/requirements_OpenAI.txt")
llm_judge_evaluate_homogeneity_viewpoint_text_embedding_005_KMEANS_studio.run("python 6001_llm_judge_evaluate_homogeneity.py \
    --inputfile /teamspace/studios/5002-cluster-embeddings-kmeans/clusters/memes_stage5002A_completed.csv \
    --clustercol cluster \
    --outputfile /teamspace/studios/6001-llm-judge-evaluate-homogeneity/evaluations/memes_stage6001A2_completed.csv \
    --inputprompts /teamspace/uploads/prompts_homogeneity_evaluation.json \
    --inputcred /teamspace/uploads/OPENAI_CREDENTIALS.env")
llm_judge_evaluate_homogeneity_viewpoint_text_embedding_005_KMEANS_studio.stop()
llm_judge_evaluate_homogeneity_viewpoint_text_embedding_005_KMEANS_stop = perf_counter()
llm_judge_evaluate_homogeneity_viewpoint_text_embedding_005_KMEANS_time = llm_judge_evaluate_homogeneity_viewpoint_text_embedding_005_KMEANS_stop - llm_judge_evaluate_homogeneity_viewpoint_text_embedding_005_KMEANS_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "llm_judge_evaluate_homogeneity_viewpoint_text_embedding_005_KMEANS_time": str(llm_judge_evaluate_homogeneity_viewpoint_text_embedding_005_KMEANS_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="llm_judge_evaluate_homogeneity_viewpoint_text_embedding_005_KMEANS_time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)