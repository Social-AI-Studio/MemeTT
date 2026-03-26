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
mistral_small_2503_inference_fix_start = perf_counter()
mistral_small_2503_inference_fix_studio = Studio("2006_mistral_small_2503_inference_fix")
mistral_small_2503_inference_fix_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
mistral_small_2503_inference_fix_studio.run("python -m pip install --upgrade pip")
mistral_small_2503_inference_fix_studio.run("pip install -r /teamspace/uploads/requirements_MISTRALAI.txt")
mistral_small_2503_inference_fix_studio.run("python 2006_mistral_small_2503_inference_fix.py \
    --inputfile /teamspace/studios/2006-mistral-small-2503-inference/inferences/memes_stage2006_completed.csv \
    --outputfiletemp /teamspace/studios/2006-mistral-small-2503-inference-fix/inferences/memes_stage2006_completed_fix_temp.csv \
    --outputfile /teamspace/studios/2006-mistral-small-2503-inference-fix/inferences/memes_stage2006_completed_fix.csv \
    --inputprompts /teamspace/uploads/prompts_mabsa.json \
    --inputcred /teamspace/uploads/MISTRAL_CREDENTIALS.env")
mistral_small_2503_inference_fix_studio.stop()
mistral_small_2503_inference_fix_stop = perf_counter()
mistral_small_2503_inference_fix_time = mistral_small_2503_inference_fix_stop - mistral_small_2503_inference_fix_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "mistral_small_2503_inference_fix_time": str(mistral_small_2503_inference_fix_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)