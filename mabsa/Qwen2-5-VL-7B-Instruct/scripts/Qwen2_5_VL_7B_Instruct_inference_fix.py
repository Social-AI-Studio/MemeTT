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
Qwen2_5_VL_7B_Instruct_inference_fix_start = perf_counter()
Qwen2_5_VL_7B_Instruct_inference_fix_studio = Studio("2010_Qwen2_5_VL_7B_Instruct_inference_fix")
Qwen2_5_VL_7B_Instruct_inference_fix_studio.start(Machine.L40S)
time.sleep(100)
Qwen2_5_VL_7B_Instruct_inference_fix_studio.run("python -m pip install --upgrade pip")
Qwen2_5_VL_7B_Instruct_inference_fix_studio.run("pip install -r /teamspace/uploads/requirements_VLLM.txt")
Qwen2_5_VL_7B_Instruct_inference_fix_studio.run('(cd "$(mktemp -d)" && git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention && pip install .)')
Qwen2_5_VL_7B_Instruct_inference_fix_studio.run("python 2010_Qwen2_5_VL_7B_Instruct_inference_fix.py \
    --inputfile /teamspace/studios/2010-qwen2-5-vl-7b-instruct-inference/inferences/memes_stage2010_completed.csv \
    --outputfiletemp /teamspace/studios/2010-qwen2-5-vl-7b-instruct-inference-fix/inferences/memes_stage2010_completed_fix_temp.csv \
    --outputfile /teamspace/studios/2010-qwen2-5-vl-7b-instruct-inference-fix/inferences/memes_stage2010_completed_fix.csv \
    --inputprompts /teamspace/uploads/prompts_mabsa.json")
Qwen2_5_VL_7B_Instruct_inference_fix_studio.stop()
Qwen2_5_VL_7B_Instruct_inference_fix_stop = perf_counter()
Qwen2_5_VL_7B_Instruct_inference_fix_time = Qwen2_5_VL_7B_Instruct_inference_fix_stop - Qwen2_5_VL_7B_Instruct_inference_fix_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "Qwen2_5_VL_7B_Instruct_inference_fix_time": str(Qwen2_5_VL_7B_Instruct_inference_fix_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)