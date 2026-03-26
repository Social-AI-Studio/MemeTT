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
InternVL3_8B_hf_extract_quintuplets_start = perf_counter()
InternVL3_8B_hf_extract_quintuplets_studio = Studio("3000_extract_quintuplets")
InternVL3_8B_hf_extract_quintuplets_studio.start(Machine.DATA_PREP)
time.sleep(100)
InternVL3_8B_hf_extract_quintuplets_studio.run("python -m pip install --upgrade pip")
InternVL3_8B_hf_extract_quintuplets_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
InternVL3_8B_hf_extract_quintuplets_studio.run("python 3000_extract_quintuplets.py \
    --inputfile /teamspace/studios/2011-internvl3-8b-hf-inference-fix/inferences/memes_stage2011_completed_fix.csv \
    --outputfile /teamspace/studios/3000-extract-quintuplets/inferences/memes_stage3011_completed.csv \
    --textcolname InternVL3_8B_hf_response_text \
    --inputprompts /teamspace/uploads/prompts_extract_quad.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
InternVL3_8B_hf_extract_quintuplets_studio.stop()
InternVL3_8B_hf_extract_quintuplets_stop = perf_counter()
InternVL3_8B_hf_extract_quintuplets_time = InternVL3_8B_hf_extract_quintuplets_stop - InternVL3_8B_hf_extract_quintuplets_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "InternVL3_8B_hf_extract_quintuplets_time": str(InternVL3_8B_hf_extract_quintuplets_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)