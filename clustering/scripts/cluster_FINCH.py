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
cluster_start = perf_counter()
cluster_studio = Studio("5001_cluster_embeddings_FINCH")
cluster_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
cluster_studio.run("python -m pip install --upgrade pip")
cluster_studio.run("pip install pandas==2.2.3")
cluster_studio.run("pip install numpy==2.2.5")
cluster_studio.run("pip install parallel-pandas==0.6.5")
cluster_studio.run("pip install pynndescent==0.5.13")
cluster_studio.run("pip install git+https://github.com/ssarfraz/FINCH-Clustering.git")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4000_completed.csv \
    --embeddingscol viewpoint_embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001A_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4001_completed.csv \
    --embeddingscol viewpoint_embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001B_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4002_completed.csv \
    --embeddingscol viewpoint_embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001C_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4003_completed.csv \
    --embeddingscol viewpoint_embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001D_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4004_completed.csv \
    --embeddingscol viewpoint_embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001E_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4005_completed.csv \
    --embeddingscol viewpoint_embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001F_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4006_completed.csv \
    --embeddingscol viewpoint_embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001G_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4007_completed.csv \
    --embeddingscol viewpoint_embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001H_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4008_completed.csv \
    --embeddingscol viewpoint_embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001I_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4009_completed.csv \
    --embeddingscol viewpoint_embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001J_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4010_completed.csv \
    --embeddingscol viewpoint_embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001K_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4011_completed.csv \
    --embeddingscol viewpoint_embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001L_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4001-get-embeddings-text-text-embedding-005/embeddings/memes_stage4001_completed.csv \
    --embeddingscol text_embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001M_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4002-get-embeddings-text-embed-v4-0/embeddings/memes_stage4002_completed.csv \
    --embeddingscol text_embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001N_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4003-get-embeddings-mixed-embed-v4-0/embeddings/memes_stage4003_completed.csv \
    --embeddingscol embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001O_completed.csv")
cluster_studio.run("python 5001_cluster_embeddings_FINCH.py \
    --inputfile /teamspace/studios/4004-get-embeddings-voyage-multimodal-3/embeddings/memes_stage4004_completed.csv \
    --embeddingscol embeddings \
    --outputfile /teamspace/studios/5001-cluster-embeddings-finch/clusters/memes_stage5001P_completed.csv")
cluster_studio.stop()
cluster_stop = perf_counter()
cluster_time = cluster_stop - cluster_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "cluster_time": str(cluster_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)