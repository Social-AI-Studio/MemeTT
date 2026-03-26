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
llm_judge_evaluate_intracluster_quality_start = perf_counter()
llm_judge_evaluate_intracluster_quality_studio = Studio("6002_llm_judge_evaluate_intracluster_quality")
llm_judge_evaluate_intracluster_quality_studio.start(Machine.DATA_PREP)
time.sleep(100)
llm_judge_evaluate_intracluster_quality_studio.run("python -m pip install --upgrade pip")
llm_judge_evaluate_intracluster_quality_studio.run("pip install -r /teamspace/uploads/requirements_OpenAI.txt")
llm_judge_evaluate_intracluster_quality_studio.run("python 6002_llm_judge_evaluate_intracluster_quality.py \
    --inputcoherence coherence.csv \
    --inputrelevance relevance.csv \
    --outputcoherenceeval /teamspace/studios/6002-llm-judge-evaluate-intracluster-quality/evaluations/memes_stage6002_coherence_completed.csv \
    --outputrelevanceeval /teamspace/studios/6002-llm-judge-evaluate-intracluster-quality/evaluations/memes_stage6002_relevance_completed.csv \
    --inputprompts /teamspace/uploads/prompts_intracluster_quality_evaluation.json \
    --inputcred /teamspace/uploads/OPENAI_CREDENTIALS.env")
llm_judge_evaluate_intracluster_quality_studio.stop()
llm_judge_evaluate_intracluster_quality_stop = perf_counter()
llm_judge_evaluate_intracluster_quality_time = llm_judge_evaluate_intracluster_quality_stop - llm_judge_evaluate_intracluster_quality_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "llm_judge_evaluate_intracluster_quality_time": str(llm_judge_evaluate_intracluster_quality_time),
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="llm_judge_evaluate_intracluster_quality_time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)