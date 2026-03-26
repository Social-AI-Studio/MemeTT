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
llama4_scout_instruct_basic_extract_quintuplets_start = perf_counter()
llama4_scout_instruct_basic_extract_quintuplets_studio = Studio("3000_extract_quintuplets")
llama4_scout_instruct_basic_extract_quintuplets_studio.start(Machine.DATA_PREP)
time.sleep(100)
llama4_scout_instruct_basic_extract_quintuplets_studio.run("python -m pip install --upgrade pip")
llama4_scout_instruct_basic_extract_quintuplets_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
llama4_scout_instruct_basic_extract_quintuplets_studio.run("python 3000_extract_quintuplets.py \
    --inputfile /teamspace/studios/2004-llama4-scout-instruct-basic-inference-fix/inferences/memes_stage2004_completed_fix.csv \
    --outputfile /teamspace/studios/3000-extract-quintuplets/inferences/memes_stage3004_completed.csv \
    --textcolname llama4_scout_instruct_basic_response_text \
    --inputprompts /teamspace/uploads/prompts_extract_quad.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
llama4_scout_instruct_basic_extract_quintuplets_studio.stop()
llama4_scout_instruct_basic_extract_quintuplets_stop = perf_counter()
llama4_scout_instruct_basic_extract_quintuplets_time = llama4_scout_instruct_basic_extract_quintuplets_stop - llama4_scout_instruct_basic_extract_quintuplets_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "llama4_scout_instruct_basic_extract_quintuplets_time": str(llama4_scout_instruct_basic_extract_quintuplets_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)