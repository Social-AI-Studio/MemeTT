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
llama4_scout_instruct_basic_inference_fix_start = perf_counter()
llama4_scout_instruct_basic_inference_fix_studio = Studio("2004_llama4_scout_instruct_basic_inference_fix")
llama4_scout_instruct_basic_inference_fix_studio.start(Machine.DATA_PREP)
time.sleep(100)
llama4_scout_instruct_basic_inference_fix_studio.run("python -m pip install --upgrade pip")
llama4_scout_instruct_basic_inference_fix_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
llama4_scout_instruct_basic_inference_fix_studio.run("python 2004_llama4_scout_instruct_basic_inference_fix.py \
    --inputfile /teamspace/studios/2004-llama4-scout-instruct-basic-inference/inferences/memes_stage2004_completed.csv \
    --outputfiletemp /teamspace/studios/2004-llama4-scout-instruct-basic-inference-fix/inferences/memes_stage2004_completed_fix_temp.csv \
    --outputfile /teamspace/studios/2004-llama4-scout-instruct-basic-inference-fix/inferences/memes_stage2004_completed_fix.csv \
    --inputprompts /teamspace/uploads/prompts_mabsa.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
llama4_scout_instruct_basic_inference_fix_studio.stop()
llama4_scout_instruct_basic_inference_fix_stop = perf_counter()
llama4_scout_instruct_basic_inference_fix_time = llama4_scout_instruct_basic_inference_fix_stop - llama4_scout_instruct_basic_inference_fix_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "llama4_scout_instruct_basic_inference_fix_time": str(llama4_scout_instruct_basic_inference_fix_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)