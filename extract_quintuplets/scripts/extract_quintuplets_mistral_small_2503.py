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
mistral_small_2503_extract_quintuplets_start = perf_counter()
mistral_small_2503_extract_quintuplets_studio = Studio("3000_extract_quintuplets")
mistral_small_2503_extract_quintuplets_studio.start(Machine.DATA_PREP)
time.sleep(100)
mistral_small_2503_extract_quintuplets_studio.run("python -m pip install --upgrade pip")
mistral_small_2503_extract_quintuplets_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
mistral_small_2503_extract_quintuplets_studio.run("python 3000_extract_quintuplets.py \
    --inputfile /teamspace/studios/2006-mistral-small-2503-inference-fix/inferences/memes_stage2006_completed_fix.csv \
    --outputfile /teamspace/studios/3000-extract-quintuplets/inferences/memes_stage3006_completed.csv \
    --textcolname mistral_small_2503_response_text \
    --inputprompts /teamspace/uploads/prompts_extract_quad.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
mistral_small_2503_extract_quintuplets_studio.stop()
mistral_small_2503_extract_quintuplets_stop = perf_counter()
mistral_small_2503_extract_quintuplets_time = mistral_small_2503_extract_quintuplets_stop - mistral_small_2503_extract_quintuplets_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "mistral_small_2503_extract_quintuplets_time": str(mistral_small_2503_extract_quintuplets_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)