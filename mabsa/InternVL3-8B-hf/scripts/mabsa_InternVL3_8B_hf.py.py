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
InternVL3_8B_hf_inference_start = perf_counter()
InternVL3_8B_hf_inference_studio = Studio("2011_InternVL3_8B_hf_inference")
InternVL3_8B_hf_inference_studio.start(Machine.L40S)
time.sleep(100)
InternVL3_8B_hf_inference_studio.run("python -m pip install --upgrade pip")
InternVL3_8B_hf_inference_studio.run("pip install -r /teamspace/uploads/requirements_INTERNVL3.txt")
InternVL3_8B_hf_inference_studio.run("pip install git+https://github.com/huggingface/transformers.git")
InternVL3_8B_hf_inference_studio.run("python 2011_InternVL3_8B_hf_inference.py \
    --inputfile /teamspace/studios/1001-recognize-describe-celebrities/recognize_describe_celebrities/memes_stage1001_completed.csv \
    --outputfile /teamspace/studios/2011-internvl3-8b-hf-inference/inferences/memes_stage2011_completed.csv \
    --inputprompts /teamspace/uploads/prompts_mabsa.json")
InternVL3_8B_hf_inference_studio.stop()
InternVL3_8B_hf_inference_stop = perf_counter()
InternVL3_8B_hf_inference_time = InternVL3_8B_hf_inference_stop - InternVL3_8B_hf_inference_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "InternVL3_8B_hf_inference_time": str(InternVL3_8B_hf_inference_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)