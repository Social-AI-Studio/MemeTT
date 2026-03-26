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
cluster_studio = Studio("5002_cluster_embeddings_KMEANS")
cluster_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
cluster_studio.run("python -m pip install --upgrade pip")
cluster_studio.run("pip install pandas==2.2.3")
cluster_studio.run("pip install numpy==1.26.4")
cluster_studio.run("pip install parallel-pandas==0.6.5")
cluster_studio.run("pip install scikit-learn==1.8.0")
cluster_studio.run("python 5002_cluster_embeddings_KMEANS.py \
    --inputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4000_completed.csv \
    --embeddingscol viewpoint_embeddings \
    --outputfile /teamspace/studios/5002-cluster-embeddings-kmeans/clusters/memes_stage5002A_completed.csv \
    --n_clusters 2484")
cluster_studio.run("python 5002_cluster_embeddings_KMEANS.py \
    --inputfile /teamspace/studios/4001-get-embeddings-text-text-embedding-005/embeddings/memes_stage4001_completed.csv \
    --embeddingscol text_embeddings \
    --outputfile /teamspace/studios/5002-cluster-embeddings-kmeans/clusters/memes_stage5002M_completed.csv \
    --n_clusters 1701")
cluster_studio.run("python 5002_cluster_embeddings_KMEANS.py \
    --inputfile /teamspace/studios/4002-get-embeddings-text-embed-v4-0/embeddings/memes_stage4002_completed.csv \
    --embeddingscol text_embeddings \
    --outputfile /teamspace/studios/5002-cluster-embeddings-kmeans/clusters/memes_stage5002N_completed.csv \
    --n_clusters 1604")
cluster_studio.run("python 5002_cluster_embeddings_KMEANS.py \
    --inputfile /teamspace/studios/4003-get-embeddings-mixed-embed-v4-0/embeddings/memes_stage4003_completed.csv \
    --embeddingscol embeddings \
    --outputfile /teamspace/studios/5002-cluster-embeddings-kmeans/clusters/memes_stage5002O_completed.csv \
    --n_clusters 1676")
cluster_studio.run("python 5002_cluster_embeddings_KMEANS.py \
    --inputfile /teamspace/studios/4004-get-embeddings-voyage-multimodal-3/embeddings/memes_stage4004_completed.csv \
    --embeddingscol embeddings \
    --outputfile /teamspace/studios/5002-cluster-embeddings-kmeans/clusters/memes_stage5002P_completed.csv \
    --n_clusters 1736")
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