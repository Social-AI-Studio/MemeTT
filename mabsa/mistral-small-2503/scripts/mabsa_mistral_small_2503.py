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
mistral_small_2503_inference_start = perf_counter()
mistral_small_2503_inference_studio = Studio("2006_mistral_small_2503_inference")
mistral_small_2503_inference_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
mistral_small_2503_inference_studio.run("python -m pip install --upgrade pip")
mistral_small_2503_inference_studio.run("pip install -r /teamspace/uploads/requirements_MISTRALAI.txt")
mistral_small_2503_inference_studio.run("python 2006_mistral_small_2503_inference.py \
    --inputfile /teamspace/studios/1001-recognize-describe-celebrities/recognize_describe_celebrities/memes_stage1001_completed.csv \
    --outputfile /teamspace/studios/2006-mistral-small-2503-inference/inferences/memes_stage2006_completed.csv \
    --inputprompts /teamspace/uploads/prompts_mabsa.json \
    --inputcred /teamspace/uploads/MISTRAL_CREDENTIALS.env")
mistral_small_2503_inference_studio.stop()
mistral_small_2503_inference_stop = perf_counter()
mistral_small_2503_inference_time = mistral_small_2503_inference_stop - mistral_small_2503_inference_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "mistral_small_2503_inference_time": str(mistral_small_2503_inference_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)