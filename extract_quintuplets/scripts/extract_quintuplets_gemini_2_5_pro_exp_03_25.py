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
gemini_2_5_pro_exp_03_25_extract_quintuplets_start = perf_counter()
gemini_2_5_pro_exp_03_25_extract_quintuplets_studio = Studio("3000_extract_quintuplets")
gemini_2_5_pro_exp_03_25_extract_quintuplets_studio.start(Machine.DATA_PREP)
time.sleep(100)
gemini_2_5_pro_exp_03_25_extract_quintuplets_studio.run("python -m pip install --upgrade pip")
gemini_2_5_pro_exp_03_25_extract_quintuplets_studio.run("pip install -r /teamspace/uploads/requirements_FIREWORKS.txt")
gemini_2_5_pro_exp_03_25_extract_quintuplets_studio.run("python 3000_extract_quintuplets.py \
    --inputfile /teamspace/studios/2000-gemini-2-5-pro-exp-03-25-inference-fix/inferences/memes_stage2000_completed_fix.csv \
    --outputfile /teamspace/studios/3000-extract-quintuplets/inferences/memes_stage3000_completed.csv \
    --textcolname gemini_2_5_pro_exp_03_25_response_text \
    --inputprompts /teamspace/uploads/prompts_extract_quad.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env")
gemini_2_5_pro_exp_03_25_extract_quintuplets_studio.stop()
gemini_2_5_pro_exp_03_25_extract_quintuplets_stop = perf_counter()
gemini_2_5_pro_exp_03_25_extract_quintuplets_time = gemini_2_5_pro_exp_03_25_extract_quintuplets_stop - gemini_2_5_pro_exp_03_25_extract_quintuplets_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "gemini_2_5_pro_exp_03_25_extract_quintuplets_time": str(gemini_2_5_pro_exp_03_25_extract_quintuplets_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)