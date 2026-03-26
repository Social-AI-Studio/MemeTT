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
Qwen2_5_VL_7B_Instruct_get_embeddings_start = perf_counter()
Qwen2_5_VL_7B_Instruct_get_embeddings_studio = Studio("4000_get_embeddings")
Qwen2_5_VL_7B_Instruct_get_embeddings_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
Qwen2_5_VL_7B_Instruct_get_embeddings_studio.run("python -m pip install --upgrade pip")
Qwen2_5_VL_7B_Instruct_get_embeddings_studio.run("pip install -r /teamspace/uploads/requirements_VERTEXAI.txt")
Qwen2_5_VL_7B_Instruct_get_embeddings_studio.run("python 4000_get_embeddings.py \
    --inputfile /teamspace/studios/3000-extract-quintuplets/inferences/memes_stage3010_completed.csv \
    --outputfileunique /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4010_completed_unique.csv \
    --outputfile /teamspace/studios/4000-get-embeddings/embeddings/memes_stage4010_completed.csv \
    --inputcred /teamspace/uploads/GOOGLE_APPLICATION_CREDENTIALS_VERTEX.json \
    --projectdetails /teamspace/uploads/GOOGLE_CLOUD_PROJECT_DETAILS.env")
Qwen2_5_VL_7B_Instruct_get_embeddings_studio.stop()
Qwen2_5_VL_7B_Instruct_get_embeddings_stop = perf_counter()
Qwen2_5_VL_7B_Instruct_get_embeddings_time = Qwen2_5_VL_7B_Instruct_get_embeddings_stop - Qwen2_5_VL_7B_Instruct_get_embeddings_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "Qwen2_5_VL_7B_Instruct_get_embeddings_time": str(Qwen2_5_VL_7B_Instruct_get_embeddings_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)