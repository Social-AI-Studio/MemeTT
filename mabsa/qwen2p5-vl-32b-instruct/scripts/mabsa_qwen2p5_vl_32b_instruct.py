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
qwen2p5_vl_32b_instruct_inference_start = perf_counter()
qwen2p5_vl_32b_instruct_inference_studio = Studio("2008_qwen2p5_vl_32b_instruct_inference")
qwen2p5_vl_32b_instruct_inference_studio.start(Machine.DATA_PREP)
time.sleep(100)
qwen2p5_vl_32b_instruct_inference_studio.run("python -m pip install --upgrade pip")
qwen2p5_vl_32b_instruct_inference_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
qwen2p5_vl_32b_instruct_inference_studio.run("python 2008_qwen2p5_vl_32b_instruct_inference.py \
    --inputfile /teamspace/studios/1001-recognize-describe-celebrities/recognize_describe_celebrities/memes_stage1001_completed.csv \
    --outputfile /teamspace/studios/2008-qwen2p5-vl-32b-instruct-inference/inferences/memes_stage2008_completed.csv \
    --inputprompts /teamspace/uploads/prompts_mabsa.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
qwen2p5_vl_32b_instruct_inference_studio.stop()
qwen2p5_vl_32b_instruct_inference_stop = perf_counter()
qwen2p5_vl_32b_instruct_inference_time = qwen2p5_vl_32b_instruct_inference_stop - qwen2p5_vl_32b_instruct_inference_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "qwen2p5_vl_32b_instruct_inference_time": str(qwen2p5_vl_32b_instruct_inference_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)