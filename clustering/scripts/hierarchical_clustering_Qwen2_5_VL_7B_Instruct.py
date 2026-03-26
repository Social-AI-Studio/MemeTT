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
Qwen2_5_VL_7B_Instruct_hierarchical_cluster_start = perf_counter()
Qwen2_5_VL_7B_Instruct_hierarchical_cluster_studio = Studio("5000_hierarchical_clustering")
Qwen2_5_VL_7B_Instruct_hierarchical_cluster_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
Qwen2_5_VL_7B_Instruct_hierarchical_cluster_studio.run("python -m pip install --upgrade pip")
Qwen2_5_VL_7B_Instruct_hierarchical_cluster_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
Qwen2_5_VL_7B_Instruct_hierarchical_cluster_studio.run("pip install pynndescent==0.5.13")
Qwen2_5_VL_7B_Instruct_hierarchical_cluster_studio.run("pip install json-repair==0.44.1")
Qwen2_5_VL_7B_Instruct_hierarchical_cluster_studio.run("pip install git+https://github.com/ssarfraz/FINCH-Clustering.git")
Qwen2_5_VL_7B_Instruct_hierarchical_cluster_studio.run("python 5000_hierarchical_clustering.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4010_completed.csv \
    --outputfile /teamspace/studios/5000-hierarchical-clustering/clusters/memes_stage5010_completed.csv \
    --outputfilesubset /teamspace/studios/5000-hierarchical-clustering/clusters/memes_stage5010_completed_subset.csv \
    --inputprompts /teamspace/uploads/prompts_cluster_label_narrative.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
Qwen2_5_VL_7B_Instruct_hierarchical_cluster_studio.stop()
Qwen2_5_VL_7B_Instruct_hierarchical_cluster_stop = perf_counter()
Qwen2_5_VL_7B_Instruct_hierarchical_cluster_time = Qwen2_5_VL_7B_Instruct_hierarchical_cluster_stop - Qwen2_5_VL_7B_Instruct_hierarchical_cluster_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "Qwen2_5_VL_7B_Instruct_hierarchical_cluster_time": str(Qwen2_5_VL_7B_Instruct_hierarchical_cluster_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)