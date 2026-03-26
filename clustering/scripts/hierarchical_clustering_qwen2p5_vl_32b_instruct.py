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
qwen2p5_vl_32b_instruct_hierarchical_cluster_start = perf_counter()
qwen2p5_vl_32b_instruct_hierarchical_cluster_studio = Studio("5000_hierarchical_clustering")
qwen2p5_vl_32b_instruct_hierarchical_cluster_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
qwen2p5_vl_32b_instruct_hierarchical_cluster_studio.run("python -m pip install --upgrade pip")
qwen2p5_vl_32b_instruct_hierarchical_cluster_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
qwen2p5_vl_32b_instruct_hierarchical_cluster_studio.run("pip install pynndescent==0.5.13")
qwen2p5_vl_32b_instruct_hierarchical_cluster_studio.run("pip install json-repair==0.44.1")
qwen2p5_vl_32b_instruct_hierarchical_cluster_studio.run("pip install git+https://github.com/ssarfraz/FINCH-Clustering.git")
qwen2p5_vl_32b_instruct_hierarchical_cluster_studio.run("python 5000_hierarchical_clustering.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4008_completed.csv \
    --outputfile /teamspace/studios/5000-hierarchical-clustering/clusters/memes_stage5008_completed.csv \
    --outputfilesubset /teamspace/studios/5000-hierarchical-clustering/clusters/memes_stage5008_completed_subset.csv \
    --inputprompts /teamspace/uploads/prompts_cluster_label_narrative.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
qwen2p5_vl_32b_instruct_hierarchical_cluster_studio.stop()
qwen2p5_vl_32b_instruct_hierarchical_cluster_stop = perf_counter()
qwen2p5_vl_32b_instruct_hierarchical_cluster_time = qwen2p5_vl_32b_instruct_hierarchical_cluster_stop - qwen2p5_vl_32b_instruct_hierarchical_cluster_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "qwen2p5_vl_32b_instruct_hierarchical_cluster_time": str(qwen2p5_vl_32b_instruct_hierarchical_cluster_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)