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
pixtral_12b_2409_inference_start = perf_counter()
pixtral_12b_2409_inference_studio = Studio("2007_pixtral_12b_2409_inference")
pixtral_12b_2409_inference_studio.start(Machine.DATA_PREP_MAX)
time.sleep(100)
pixtral_12b_2409_inference_studio.run("python -m pip install --upgrade pip")
pixtral_12b_2409_inference_studio.run("pip install -r /teamspace/uploads/requirements_MISTRALAI.txt")
pixtral_12b_2409_inference_studio.run("python 2007_pixtral_12b_2409_inference.py \
    --inputfile /teamspace/studios/1001-recognize-describe-celebrities/recognize_describe_celebrities/memes_stage1001_completed.csv \
    --outputfile /teamspace/studios/2007-pixtral-12b-2409-inference/inferences/memes_stage2007_completed.csv \
    --inputprompts /teamspace/uploads/prompts_mabsa.json \
    --inputcred /teamspace/uploads/MISTRAL_CREDENTIALS.env")
pixtral_12b_2409_inference_studio.stop()
pixtral_12b_2409_inference_stop = perf_counter()
pixtral_12b_2409_inference_time = pixtral_12b_2409_inference_stop - pixtral_12b_2409_inference_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "pixtral_12b_2409_inference_time": str(pixtral_12b_2409_inference_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)