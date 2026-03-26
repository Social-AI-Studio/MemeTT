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
qwen2p5_vl_32b_instruct_inference_fix_start = perf_counter()
qwen2p5_vl_32b_instruct_inference_fix_studio = Studio("2008_qwen2p5_vl_32b_instruct_inference_fix")
qwen2p5_vl_32b_instruct_inference_fix_studio.start(Machine.DATA_PREP)
time.sleep(100)
qwen2p5_vl_32b_instruct_inference_fix_studio.run("python -m pip install --upgrade pip")
qwen2p5_vl_32b_instruct_inference_fix_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
qwen2p5_vl_32b_instruct_inference_fix_studio.run("python 2008_qwen2p5_vl_32b_instruct_inference_fix.py \
    --inputfile /teamspace/studios/2008-qwen2p5-vl-32b-instruct-inference/inferences/memes_stage2008_completed.csv \
    --outputfiletemp /teamspace/studios/2008-qwen2p5-vl-32b-instruct-inference-fix/inferences/memes_stage2008_completed_fix_temp.csv \
    --outputfile /teamspace/studios/2008-qwen2p5-vl-32b-instruct-inference-fix/inferences/memes_stage2008_completed_fix.csv \
    --inputprompts /teamspace/uploads/prompts_mabsa.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
qwen2p5_vl_32b_instruct_inference_fix_studio.stop()
qwen2p5_vl_32b_instruct_inference_fix_stop = perf_counter()
qwen2p5_vl_32b_instruct_inference_fix_time = qwen2p5_vl_32b_instruct_inference_fix_stop - qwen2p5_vl_32b_instruct_inference_fix_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "qwen2p5_vl_32b_instruct_inference_fix_time": str(qwen2p5_vl_32b_instruct_inference_fix_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)