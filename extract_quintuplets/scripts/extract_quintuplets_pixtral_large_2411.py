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
pixtral_large_2411_extract_quintuplets_start = perf_counter()
pixtral_large_2411_extract_quintuplets_studio = Studio("3000_extract_quintuplets")
pixtral_large_2411_extract_quintuplets_studio.start(Machine.DATA_PREP)
time.sleep(100)
pixtral_large_2411_extract_quintuplets_studio.run("python -m pip install --upgrade pip")
pixtral_large_2411_extract_quintuplets_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
pixtral_large_2411_extract_quintuplets_studio.run("python 3000_extract_quintuplets.py \
    --inputfile /teamspace/studios/2005-pixtral-large-2411-inference-fix/inferences/memes_stage2005_completed_fix.csv \
    --outputfile /teamspace/studios/3000-extract-quintuplets/inferences/memes_stage3005_completed.csv \
    --textcolname pixtral_large_2411_response_text \
    --inputprompts /teamspace/uploads/prompts_extract_quad.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
pixtral_large_2411_extract_quintuplets_studio.stop()
pixtral_large_2411_extract_quintuplets_stop = perf_counter()
pixtral_large_2411_extract_quintuplets_time = pixtral_large_2411_extract_quintuplets_stop - pixtral_large_2411_extract_quintuplets_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "pixtral_large_2411_extract_quintuplets_time": str(pixtral_large_2411_extract_quintuplets_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)