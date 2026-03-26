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
totaldefmeme_prep_start = perf_counter()
totaldefmeme_prep_studio = Studio("0000_TotalDefMeme_preparation")
totaldefmeme_prep_studio.start(Machine.DATA_PREP)
time.sleep(100)
totaldefmeme_prep_studio.run("python -m pip install --upgrade pip")
totaldefmeme_prep_studio.run("pip install -r /teamspace/uploads/requirements_data_preparation.txt")
totaldefmeme_prep_studio.run("python 0000_TotalDefMeme_preparation.py \
    --outputdir /teamspace/studios/0000-totaldefmeme-preparation/TotalDefMeme \
    --inputcred /teamspace/uploads/AWS_CREDENTIALS.env \
    --bucketname memett-original")
totaldefmeme_prep_studio.stop()
totaldefmeme_prep_stop = perf_counter()
totaldefmeme_prep_time = totaldefmeme_prep_stop - totaldefmeme_prep_start

harmeme_prep_start = perf_counter()
harmeme_prep_studio = Studio("0001_HarMeme_preparation")
harmeme_prep_studio.start(Machine.DATA_PREP)
time.sleep(100)
harmeme_prep_studio.run("python -m pip install --upgrade pip")
harmeme_prep_studio.run("pip install -r /teamspace/uploads/requirements_data_preparation.txt")
harmeme_prep_studio.run("python 0001_HarMeme_preparation.py \
    --outputdir /teamspace/studios/0001-harmeme-preparation/HarMeme \
    --inputcred /teamspace/uploads/AWS_CREDENTIALS.env \
    --bucketname memett-original")
harmeme_prep_studio.stop()
harmeme_prep_stop = perf_counter()
harmeme_prep_time = harmeme_prep_stop - harmeme_prep_start

concat_datasets_start = perf_counter()
concat_datasets_studio = Studio("0002_concat_datasets")
concat_datasets_studio.start(Machine.CPU)
time.sleep(100)
concat_datasets_studio.run("python -m pip install --upgrade pip")
concat_datasets_studio.run("pip install pandas==2.2.3")
concat_datasets_studio.run("python 0002_concat_datasets.py \
    --inputtdef /teamspace/studios/0000-totaldefmeme-preparation/TotalDefMeme/TotalDefMeme.csv \
    --inputharmp /teamspace/studios/0001-harmeme-preparation/HarMeme/harmp.csv \
    --inputharmc /teamspace/studios/0001-harmeme-preparation/HarMeme/harmc.csv \
    --outputfile /teamspace/studios/0002-concat-datasets/concat_datasets/memes.csv")
concat_datasets_studio.stop()
concat_datasets_stop = perf_counter()
concat_datasets_time = concat_datasets_stop - concat_datasets_start

detect_text_start = perf_counter()
detect_text_studio = Studio("1000_detect_text")
detect_text_studio.start(Machine.DATA_PREP)
time.sleep(100)
detect_text_studio.run("python -m pip install --upgrade pip")
detect_text_studio.run("pip install -r /teamspace/uploads/requirements_detect_text.txt")
detect_text_studio.run("python 1000_detect_text.py \
    --inputfile /teamspace/studios/0002-concat-datasets/concat_datasets/memes.csv \
    --outputfile /teamspace/studios/1000-detect-text/detect_text/memes_stage1000_completed.csv \
    --inputcred /teamspace/uploads/GOOGLE_APPLICATION_CREDENTIALS.json")
detect_text_studio.stop()
detect_text_stop = perf_counter()
detect_text_time = detect_text_stop - detect_text_start

recognize_describe_celebrities_start = perf_counter()
recognize_describe_celebrities_studio = Studio("1001_recognize_describe_celebrities")
recognize_describe_celebrities_studio.start(Machine.DATA_PREP)
time.sleep(100)
recognize_describe_celebrities_studio.run("python -m pip install --upgrade pip")
recognize_describe_celebrities_studio.run("pip install -r /teamspace/uploads/requirements_recognize_describe_celebrities.txt")
recognize_describe_celebrities_studio.run("python 1001_recognize_describe_celebrities.py \
    --inputfile /teamspace/studios/1000-detect-text/detect_text/memes_stage1000_completed.csv \
    --outputfile /teamspace/studios/1001-recognize-describe-celebrities/recognize_describe_celebrities/memes_stage1001_completed.csv \
    --outputpath /teamspace/studios/1001-recognize-describe-celebrities/recognize_describe_celebrities/memes_draw_bboxes \
    --inputcred /teamspace/uploads/AWS_CREDENTIALS.env \
    --minconf 90 \
    --bucketnameinput memett-original \
    --bucketnamesave memett-bboxes")
recognize_describe_celebrities_studio.stop()
recognize_describe_celebrities_stop = perf_counter()
recognize_describe_celebrities_time = recognize_describe_celebrities_stop - recognize_describe_celebrities_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "totaldefmeme_prep_time": str(totaldefmeme_prep_time),
    "harmeme_prep_time": str(harmeme_prep_time),
    "concat_datasets_time": str(concat_datasets_time),
    "detect_text_time": str(detect_text_time),
    "recognize_describe_celebrities_time": str(recognize_describe_celebrities_time)
}
time_taken_json = json.dumps(time_taken_dict)
with open(file="time_taken.json", mode="w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)