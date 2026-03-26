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
gemini_2_0_flash_thinking_exp_01_21_inference_fix_start = perf_counter()
gemini_2_0_flash_thinking_exp_01_21_inference_fix_studio = Studio("2001_gemini_2_0_flash_thinking_exp_01_21_inference_fix")
gemini_2_0_flash_thinking_exp_01_21_inference_fix_studio.start(Machine.CPU)
time.sleep(100)
gemini_2_0_flash_thinking_exp_01_21_inference_fix_studio.run("python -m pip install --upgrade pip")
gemini_2_0_flash_thinking_exp_01_21_inference_fix_studio.run("pip install -r /teamspace/uploads/requirements_VERTEXAI.txt")
gemini_2_0_flash_thinking_exp_01_21_inference_fix_studio.run("python 2001_gemini_2_0_flash_thinking_exp_01_21_inference_fix.py \
    --inputfile /teamspace/studios/2001-gemini-2-0-flash-thinking-exp-01-21-inference/inferences/memes_stage2001_completed.csv \
    --outputfiletemp /teamspace/studios/2001-gemini-2-0-flash-thinking-exp-01-21-inference-fix/inferences/memes_stage2001_completed_fix_temp.csv \
    --outputfile /teamspace/studios/2001-gemini-2-0-flash-thinking-exp-01-21-inference-fix/inferences/memes_stage2001_completed_fix.csv \
    --inputprompts /teamspace/uploads/prompts_mabsa.json \
    --inputcred /teamspace/uploads/GOOGLE_APPLICATION_CREDENTIALS_VERTEX.json \
    --projectdetails /teamspace/uploads/GOOGLE_CLOUD_PROJECT_DETAILS.env")
gemini_2_0_flash_thinking_exp_01_21_inference_fix_studio.stop()
gemini_2_0_flash_thinking_exp_01_21_inference_fix_stop = perf_counter()
gemini_2_0_flash_thinking_exp_01_21_inference_fix_time = gemini_2_0_flash_thinking_exp_01_21_inference_fix_stop - gemini_2_0_flash_thinking_exp_01_21_inference_fix_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "gemini_2_0_flash_thinking_exp_01_21_inference_fix_time": str(gemini_2_0_flash_thinking_exp_01_21_inference_fix_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)