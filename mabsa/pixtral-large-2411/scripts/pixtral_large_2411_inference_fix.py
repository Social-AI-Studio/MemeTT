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
pixtral_large_2411_inference_fix_start = perf_counter()
pixtral_large_2411_inference_fix_studio = Studio("2005_pixtral_large_2411_inference_fix")
pixtral_large_2411_inference_fix_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
pixtral_large_2411_inference_fix_studio.run("python -m pip install --upgrade pip")
pixtral_large_2411_inference_fix_studio.run("pip install -r /teamspace/uploads/requirements_MISTRALAI.txt")
pixtral_large_2411_inference_fix_studio.run("python 2005_pixtral_large_2411_inference_fix.py \
    --inputfile /teamspace/studios/2005-pixtral-large-2411-inference/inferences/memes_stage2005_completed.csv \
    --outputfiletemp /teamspace/studios/2005-pixtral-large-2411-inference-fix/inferences/memes_stage2005_completed_fix_temp.csv \
    --outputfile /teamspace/studios/2005-pixtral-large-2411-inference-fix/inferences/memes_stage2005_completed_fix.csv \
    --inputprompts /teamspace/uploads/prompts_mabsa.json \
    --inputcred /teamspace/uploads/MISTRAL_CREDENTIALS.env")
pixtral_large_2411_inference_fix_studio.stop()
pixtral_large_2411_inference_fix_stop = perf_counter()
pixtral_large_2411_inference_fix_time = pixtral_large_2411_inference_fix_stop - pixtral_large_2411_inference_fix_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "pixtral_large_2411_inference_fix_time": str(pixtral_large_2411_inference_fix_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)