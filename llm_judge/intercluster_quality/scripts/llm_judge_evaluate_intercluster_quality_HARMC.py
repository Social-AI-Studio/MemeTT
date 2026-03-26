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
llm_judge_evaluate_intercluster_quality_HARMC_start = perf_counter()
llm_judge_evaluate_intercluster_quality_HARMC_studio = Studio("6003_llm_judge_evaluate_intercluster_quality")
llm_judge_evaluate_intercluster_quality_HARMC_studio.start(Machine.DATA_PREP)
time.sleep(100)
llm_judge_evaluate_intercluster_quality_HARMC_studio.run("python -m pip install --upgrade pip")
llm_judge_evaluate_intercluster_quality_HARMC_studio.run("pip install -r /teamspace/uploads/requirements_OpenAI.txt")
llm_judge_evaluate_intercluster_quality_HARMC_studio.run("python 6003_llm_judge_evaluate_intercluster_quality.py \
    --inputfile HARMC-TRIPLES.csv \
    --outputfile /teamspace/studios/6003-llm-judge-evaluate-intercluster-quality/evaluations/HARMC_completed.csv \
    --inputprompts /teamspace/uploads/prompts_intercluster_quality_evaluation.json \
    --inputcred /teamspace/uploads/OPENAI_CREDENTIALS.env")
llm_judge_evaluate_intercluster_quality_HARMC_studio.stop()
llm_judge_evaluate_intercluster_quality_HARMC_stop = perf_counter()
llm_judge_evaluate_intercluster_quality_HARMC_time = llm_judge_evaluate_intercluster_quality_HARMC_stop - llm_judge_evaluate_intercluster_quality_HARMC_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "llm_judge_evaluate_intercluster_quality_HARMC_time": str(llm_judge_evaluate_intercluster_quality_HARMC_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="llm_judge_evaluate_intercluster_quality_HARMC_time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)