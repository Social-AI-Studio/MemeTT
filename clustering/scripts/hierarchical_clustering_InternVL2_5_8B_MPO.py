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
InternVL2_5_8B_MPO_hierarchical_cluster_start = perf_counter()
InternVL2_5_8B_MPO_hierarchical_cluster_studio = Studio("5000_hierarchical_clustering")
InternVL2_5_8B_MPO_hierarchical_cluster_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
InternVL2_5_8B_MPO_hierarchical_cluster_studio.run("python -m pip install --upgrade pip")
InternVL2_5_8B_MPO_hierarchical_cluster_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
InternVL2_5_8B_MPO_hierarchical_cluster_studio.run("pip install pynndescent==0.5.13")
InternVL2_5_8B_MPO_hierarchical_cluster_studio.run("pip install json-repair==0.44.1")
InternVL2_5_8B_MPO_hierarchical_cluster_studio.run("pip install git+https://github.com/ssarfraz/FINCH-Clustering.git")
InternVL2_5_8B_MPO_hierarchical_cluster_studio.run("python 5000_hierarchical_clustering.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4009_completed.csv \
    --outputfile /teamspace/studios/5000-hierarchical-clustering/clusters/memes_stage5009_completed.csv \
    --outputfilesubset /teamspace/studios/5000-hierarchical-clustering/clusters/memes_stage5009_completed_subset.csv \
    --inputprompts /teamspace/uploads/prompts_cluster_label_narrative.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
InternVL2_5_8B_MPO_hierarchical_cluster_studio.stop()
InternVL2_5_8B_MPO_hierarchical_cluster_stop = perf_counter()
InternVL2_5_8B_MPO_hierarchical_cluster_time = InternVL2_5_8B_MPO_hierarchical_cluster_stop - InternVL2_5_8B_MPO_hierarchical_cluster_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "InternVL2_5_8B_MPO_hierarchical_cluster_time": str(InternVL2_5_8B_MPO_hierarchical_cluster_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)