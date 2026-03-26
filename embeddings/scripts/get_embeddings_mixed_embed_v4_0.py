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
get_embeddings_mixed_embed_v4_0_start = perf_counter()
get_embeddings_mixed_embed_v4_0_studio = Studio("4003_get_embeddings_mixed_embed_v4_0")
get_embeddings_mixed_embed_v4_0_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
get_embeddings_mixed_embed_v4_0_studio.run("python -m pip install --upgrade pip")
get_embeddings_mixed_embed_v4_0_studio.run("pip install -r /teamspace/uploads/requirements_COHERE.txt")
get_embeddings_mixed_embed_v4_0_studio.run("python 4003_get_embeddings_mixed_embed_v4_0.py \
    --inputfile /teamspace/studios/1001-recognize-describe-celebrities/recognize_describe_celebrities/memes_stage1001_completed.csv \
    --outputfile /teamspace/studios/4003-get-embeddings-mixed-embed-v4-0/embeddings/memes_stage4003_completed.csv \
    --inputcred /teamspace/uploads/COHERE_CREDENTIALS.env")
get_embeddings_mixed_embed_v4_0_studio.stop()
get_embeddings_mixed_embed_v4_0_stop = perf_counter()
get_embeddings_mixed_embed_v4_0_time = get_embeddings_mixed_embed_v4_0_stop - get_embeddings_mixed_embed_v4_0_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "get_embeddings_mixed_embed_v4_0_time": str(get_embeddings_mixed_embed_v4_0_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)