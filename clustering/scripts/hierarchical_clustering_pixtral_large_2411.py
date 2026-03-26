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
pixtral_large_2411_hierarchical_cluster_start = perf_counter()
pixtral_large_2411_hierarchical_cluster_studio = Studio("5000_hierarchical_clustering")
pixtral_large_2411_hierarchical_cluster_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
pixtral_large_2411_hierarchical_cluster_studio.run("python -m pip install --upgrade pip")
pixtral_large_2411_hierarchical_cluster_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
pixtral_large_2411_hierarchical_cluster_studio.run("pip install pynndescent==0.5.13")
pixtral_large_2411_hierarchical_cluster_studio.run("pip install json-repair==0.44.1")
pixtral_large_2411_hierarchical_cluster_studio.run("pip install git+https://github.com/ssarfraz/FINCH-Clustering.git")
pixtral_large_2411_hierarchical_cluster_studio.run("python 5000_hierarchical_clustering.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4005_completed.csv \
    --outputfile /teamspace/studios/5000-hierarchical-clustering/clusters/memes_stage5005_completed.csv \
    --outputfilesubset /teamspace/studios/5000-hierarchical-clustering/clusters/memes_stage5005_completed_subset.csv \
    --inputprompts /teamspace/uploads/prompts_cluster_label_narrative.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
pixtral_large_2411_hierarchical_cluster_studio.stop()
pixtral_large_2411_hierarchical_cluster_stop = perf_counter()
pixtral_large_2411_hierarchical_cluster_time = pixtral_large_2411_hierarchical_cluster_stop - pixtral_large_2411_hierarchical_cluster_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "pixtral_large_2411_hierarchical_cluster_time": str(pixtral_large_2411_hierarchical_cluster_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)