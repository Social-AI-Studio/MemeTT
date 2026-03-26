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
InternVL3_8B_hf_get_embeddings_start = perf_counter()
InternVL3_8B_hf_get_embeddings_studio = Studio("4000_get_embeddings")
InternVL3_8B_hf_get_embeddings_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
InternVL3_8B_hf_get_embeddings_studio.run("python -m pip install --upgrade pip")
InternVL3_8B_hf_get_embeddings_studio.run("pip install -r /teamspace/uploads/requirements_VERTEXAI.txt")
InternVL3_8B_hf_get_embeddings_studio.run("python 4000_get_embeddings.py \
    --inputfile /teamspace/studios/3000-extract-quintuplets/inferences/memes_stage3011_completed.csv \
    --outputfileunique /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4011_completed_unique.csv \
    --outputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4011_completed.csv \
    --inputcred /teamspace/uploads/GOOGLE_APPLICATION_CREDENTIALS_VERTEX.json \
    --projectdetails /teamspace/uploads/GOOGLE_CLOUD_PROJECT_DETAILS.env")
InternVL3_8B_hf_get_embeddings_studio.stop()
InternVL3_8B_hf_get_embeddings_stop = perf_counter()
InternVL3_8B_hf_get_embeddings_time = InternVL3_8B_hf_get_embeddings_stop - InternVL3_8B_hf_get_embeddings_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "InternVL3_8B_hf_get_embeddings_time": str(InternVL3_8B_hf_get_embeddings_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)