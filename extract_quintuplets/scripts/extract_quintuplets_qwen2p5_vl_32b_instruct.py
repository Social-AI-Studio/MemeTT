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
qwen2p5_vl_32b_instruct_extract_quintuplets_start = perf_counter()
qwen2p5_vl_32b_instruct_extract_quintuplets_studio = Studio("3000_extract_quintuplets")
qwen2p5_vl_32b_instruct_extract_quintuplets_studio.start(Machine.DATA_PREP)
time.sleep(100)
qwen2p5_vl_32b_instruct_extract_quintuplets_studio.run("python -m pip install --upgrade pip")
qwen2p5_vl_32b_instruct_extract_quintuplets_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
qwen2p5_vl_32b_instruct_extract_quintuplets_studio.run("python 3000_extract_quintuplets.py \
    --inputfile /teamspace/studios/2008-qwen2p5-vl-32b-instruct-inference-fix/inferences/memes_stage2008_completed_fix.csv \
    --outputfile /teamspace/studios/3000-extract-quintuplets/inferences/memes_stage3008_completed.csv \
    --textcolname qwen2p5_vl_32b_instruct_basic_response_text \
    --inputprompts /teamspace/uploads/prompts_extract_quad.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
qwen2p5_vl_32b_instruct_extract_quintuplets_studio.stop()
qwen2p5_vl_32b_instruct_extract_quintuplets_stop = perf_counter()
qwen2p5_vl_32b_instruct_extract_quintuplets_time = qwen2p5_vl_32b_instruct_extract_quintuplets_stop - qwen2p5_vl_32b_instruct_extract_quintuplets_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "qwen2p5_vl_32b_instruct_extract_quintuplets_time": str(qwen2p5_vl_32b_instruct_extract_quintuplets_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)