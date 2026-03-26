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
InternVL3_8B_hf_inference_fix_start = perf_counter()
InternVL3_8B_hf_inference_fix_studio = Studio("2011_InternVL3_8B_hf_inference_fix")
InternVL3_8B_hf_inference_fix_studio.start(Machine.L40S)
time.sleep(100)
InternVL3_8B_hf_inference_fix_studio.run("python -m pip install --upgrade pip")
InternVL3_8B_hf_inference_fix_studio.run("pip install -r /teamspace/uploads/requirements_INTERNVL3.txt")
InternVL3_8B_hf_inference_fix_studio.run("pip install git+https://github.com/huggingface/transformers.git")
InternVL3_8B_hf_inference_fix_studio.run("python 2011_InternVL3_8B_hf_inference_fix.py \
    --inputfile /teamspace/studios/2011-internvl3-8b-hf-inference/inferences/memes_stage2011_completed.csv \
    --outputfiletemp /teamspace/studios/2011-internvl3-8b-hf-inference-fix/inferences/memes_stage2011_completed_fix_temp.csv \
    --outputfile /teamspace/studios/2011-internvl3-8b-hf-inference-fix/inferences/memes_stage2011_completed_fix.csv \
    --inputprompts /teamspace/uploads/prompts_mabsa.json")
InternVL3_8B_hf_inference_fix_studio.stop()
InternVL3_8B_hf_inference_fix_stop = perf_counter()
InternVL3_8B_hf_inference_fix_time = InternVL3_8B_hf_inference_fix_stop - InternVL3_8B_hf_inference_fix_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "InternVL3_8B_hf_inference_fix_time": str(InternVL3_8B_hf_inference_fix_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)