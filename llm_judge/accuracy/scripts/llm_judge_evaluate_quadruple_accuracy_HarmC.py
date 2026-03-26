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
llm_judge_evaluate_quadruple_accuracy_HarmC_start = perf_counter()
llm_judge_evaluate_quadruple_accuracy_HarmC_studio = Studio("6000_llm_judge_evaluate_quadruple_accuracy")
llm_judge_evaluate_quadruple_accuracy_HarmC_studio.start(Machine.DATA_PREP)
time.sleep(100)
llm_judge_evaluate_quadruple_accuracy_HarmC_studio.run("python -m pip install --upgrade pip")
llm_judge_evaluate_quadruple_accuracy_HarmC_studio.run("pip install -r /teamspace/uploads/requirements_OpenAI.txt")
llm_judge_evaluate_quadruple_accuracy_HarmC_studio.run("python 6000_llm_judge_evaluate_quadruple_accuracy.py \
    --inputfile Harm-C.csv \
    --outputfile /teamspace/studios/6001-llm-judge-evaluate-quadruple-accuracy/evaluations/Harm-C_completed.csv \
    --inputprompts /teamspace/uploads/prompts_quadruple_accuracy_evaluation.json \
    --inputcred /teamspace/uploads/OPENAI_CREDENTIALS.env")
llm_judge_evaluate_quadruple_accuracy_HarmC_studio.stop()
llm_judge_evaluate_quadruple_accuracy_HarmC_stop = perf_counter()
llm_judge_evaluate_quadruple_accuracy_HarmC_time = llm_judge_evaluate_quadruple_accuracy_HarmC_stop - llm_judge_evaluate_quadruple_accuracy_HarmC_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "llm_judge_evaluate_quadruple_accuracy_HarmC_time": str(llm_judge_evaluate_quadruple_accuracy_HarmC_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="llm_judge_evaluate_quadruple_accuracy_HarmC_time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)