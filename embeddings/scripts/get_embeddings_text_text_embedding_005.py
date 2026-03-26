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
get_embeddings_text_text_embedding_005_start = perf_counter()
get_embeddings_text_text_embedding_005_studio = Studio("4001_get_embeddings_text_text_embedding_005")
get_embeddings_text_text_embedding_005_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
get_embeddings_text_text_embedding_005_studio.run("python -m pip install --upgrade pip")
get_embeddings_text_text_embedding_005_studio.run("pip install -r /teamspace/uploads/requirements_VERTEXAI.txt")
get_embeddings_text_text_embedding_005_studio.run("python 4001_get_embeddings_text_text_embedding_005.py \
    --inputfile /teamspace/studios/1001-recognize-describe-celebrities/recognize_describe_celebrities/memes_stage1001_completed.csv \
    --outputfileunique /teamspace/studios/4001-get-embeddings-text-text-embedding-005/embeddings/memes_stage4001_completed_unique.csv \
    --outputfile /teamspace/studios/4001-get-embeddings-text-text-embedding-005/embeddings/memes_stage4001_completed.csv \
    --inputcred /teamspace/uploads/GOOGLE_APPLICATION_CREDENTIALS_VERTEX.json \
    --projectdetails /teamspace/uploads/GOOGLE_CLOUD_PROJECT_DETAILS.env")
get_embeddings_text_text_embedding_005_studio.stop()
get_embeddings_text_text_embedding_005_stop = perf_counter()
get_embeddings_text_text_embedding_005_time = get_embeddings_text_text_embedding_005_stop - get_embeddings_text_text_embedding_005_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "get_embeddings_text_text_embedding_005_time": str(get_embeddings_text_text_embedding_005_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)