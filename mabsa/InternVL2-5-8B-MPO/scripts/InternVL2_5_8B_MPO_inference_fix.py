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
InternVL2_5_8B_MPO_inference_fix_start = perf_counter()
InternVL2_5_8B_MPO_inference_fix_studio = Studio("2009_InternVL2_5_8B_MPO_inference_fix")
InternVL2_5_8B_MPO_inference_fix_studio.start(Machine.L40S)
time.sleep(100)
InternVL2_5_8B_MPO_inference_fix_studio.run("python -m pip install --upgrade pip")
InternVL2_5_8B_MPO_inference_fix_studio.run("pip install -r /teamspace/uploads/requirements_LMDEPLOY.txt")
InternVL2_5_8B_MPO_inference_fix_studio.run('(cd "$(mktemp -d)" && git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention && pip install .)')
InternVL2_5_8B_MPO_inference_fix_studio.run("python 2009_InternVL2_5_8B_MPO_inference_fix.py \
    --inputfile /teamspace/studios/2009-internvl2-5-8b-mpo-inference/inferences/memes_stage2009_completed.csv \
    --outputfiletemp /teamspace/studios/2009-internvl2-5-8b-mpo-inference-fix/inferences/memes_stage2009_completed_fix_temp.csv \
    --outputfile /teamspace/studios/2009-internvl2-5-8b-mpo-inference-fix/inferences/memes_stage2009_completed_fix.csv \
    --inputprompts /teamspace/uploads/prompts_mabsa.json")
InternVL2_5_8B_MPO_inference_fix_studio.stop()
InternVL2_5_8B_MPO_inference_fix_stop = perf_counter()
InternVL2_5_8B_MPO_inference_fix_time = InternVL2_5_8B_MPO_inference_fix_stop - InternVL2_5_8B_MPO_inference_fix_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "InternVL2_5_8B_MPO_inference_fix_time": str(InternVL2_5_8B_MPO_inference_fix_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)