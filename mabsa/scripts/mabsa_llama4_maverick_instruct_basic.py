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
llama4_maverick_instruct_basic_inference_start = perf_counter()
llama4_maverick_instruct_basic_inference_studio = Studio("2003_llama4_maverick_instruct_basic_inference")
llama4_maverick_instruct_basic_inference_studio.start(Machine.DATA_PREP)
time.sleep(100)
llama4_maverick_instruct_basic_inference_studio.run("python -m pip install --upgrade pip")
llama4_maverick_instruct_basic_inference_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
llama4_maverick_instruct_basic_inference_studio.run("python 2003_llama4_maverick_instruct_basic_inference.py \
    --inputfile /teamspace/studios/1001-recognize-describe-celebrities/recognize_describe_celebrities/memes_stage1001_completed.csv \
    --outputfile /teamspace/studios/2003-llama4-maverick-instruct-basic-inference/inferences/memes_stage2003_completed.csv \
    --inputprompts /teamspace/uploads/prompts_mabsa.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
llama4_maverick_instruct_basic_inference_studio.stop()
llama4_maverick_instruct_basic_inference_stop = perf_counter()
llama4_maverick_instruct_basic_inference_time = llama4_maverick_instruct_basic_inference_stop - llama4_maverick_instruct_basic_inference_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "llama4_maverick_instruct_basic_inference_time": str(llama4_maverick_instruct_basic_inference_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)