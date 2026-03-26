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
Qwen2_5_VL_7B_Instruct_extract_quintuplets_start = perf_counter()
Qwen2_5_VL_7B_Instruct_extract_quintuplets_studio = Studio("3000_extract_quintuplets")
Qwen2_5_VL_7B_Instruct_extract_quintuplets_studio.start(Machine.DATA_PREP)
time.sleep(100)
Qwen2_5_VL_7B_Instruct_extract_quintuplets_studio.run("python -m pip install --upgrade pip")
Qwen2_5_VL_7B_Instruct_extract_quintuplets_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
Qwen2_5_VL_7B_Instruct_extract_quintuplets_studio.run("python 3000_extract_quintuplets.py \
    --inputfile /teamspace/studios/2010-qwen2-5-vl-7b-instruct-inference-fix/inferences/memes_stage2010_completed_fix.csv \
    --outputfile /teamspace/studios/3000-extract-quintuplets/inferences/memes_stage3010_completed.csv \
    --textcolname Qwen2_5_VL_7B_Instruct_response_text \
    --inputprompts /teamspace/uploads/prompts_extract_quad.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
Qwen2_5_VL_7B_Instruct_extract_quintuplets_studio.stop()
Qwen2_5_VL_7B_Instruct_extract_quintuplets_stop = perf_counter()
Qwen2_5_VL_7B_Instruct_extract_quintuplets_time = Qwen2_5_VL_7B_Instruct_extract_quintuplets_stop - Qwen2_5_VL_7B_Instruct_extract_quintuplets_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "Qwen2_5_VL_7B_Instruct_extract_quintuplets_time": str(Qwen2_5_VL_7B_Instruct_extract_quintuplets_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)