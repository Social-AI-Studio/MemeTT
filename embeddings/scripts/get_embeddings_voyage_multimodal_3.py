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
get_embeddings_voyage_multimodal_3_start = perf_counter()
get_embeddings_voyage_multimodal_3_studio = Studio("4004_get_embeddings_voyage_multimodal_3")
get_embeddings_voyage_multimodal_3_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
get_embeddings_voyage_multimodal_3_studio.run("python -m pip install --upgrade pip")
get_embeddings_voyage_multimodal_3_studio.run("pip install -r /teamspace/uploads/requirements_VOYAGE.txt")
get_embeddings_voyage_multimodal_3_studio.run("python 4004_get_embeddings_voyage_multimodal_3.py \
    --inputfile /teamspace/studios/1001-recognize-describe-celebrities/recognize_describe_celebrities/memes_stage1001_completed.csv \
    --outputfile /teamspace/studios/4004-get-embeddings-voyage-multimodal-3/embeddings/memes_stage4004_completed.csv \
    --inputcred /teamspace/uploads/VOYAGE_CREDENTIALS.env")
get_embeddings_voyage_multimodal_3_studio.stop()
get_embeddings_voyage_multimodal_3_stop = perf_counter()
get_embeddings_voyage_multimodal_3_time = get_embeddings_voyage_multimodal_3_stop - get_embeddings_voyage_multimodal_3_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "get_embeddings_voyage_multimodal_3_time": str(get_embeddings_voyage_multimodal_3_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)