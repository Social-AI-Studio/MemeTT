### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import json
import os
import urllib.request

import boto3
import gdown
import pandas as pd

from dotenv import load_dotenv
from parallel_pandas import ParallelPandas
from pathlib import Path
from PIL import Image
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from zipfile import ZipFile

ParallelPandas.initialize(n_cpu=32)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------Specify required command line arguments-------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description="Prepare TotalDefMeme dataset")
parser.add_argument(
    "--outputdir",
    required=True,
    help="Path to the output folder where the annotation dataset and memes will be saved"
)
parser.add_argument(
    "--inputcred",
    required=True,
    help="Absolute file path of the .env file containing AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, AWS_MAX_ATTEMPTS and AWS_RETRY_MODE"
)
parser.add_argument(
    "--bucketname",
    required=True,
    help="S3 bucket name where memes are saved"
)
args = parser.parse_args()

### --------------------------------------------------------------------------------------------------------- ###
### -------------------Load .env file containing AWS credentials, region and other parameters---------------- ###
### --------------------------------------------------------------------------------------------------------- ###
load_dotenv(Path(args.inputcred))

### --------------------------------------------------------------------------------------------------------- ###
### ----------------Create local output directory to save annotation dataset and folder of memes------------- ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(args.outputdir)
os.makedirs(
    name=relative_path,
    exist_ok=True
)

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------Download TotalDefMeme dataset------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
# Download large file from Google Drive. via
# https://github.com/wkentaro/gdown
url = "https://drive.google.com/u/0/uc?id=1oJIh4QQS3Idff2g6bZORstS5uBROjUUz"
output_path_memezip = os.path.join(
    relative_path,
    "total_def_meme.zip"
)
gdown.download(
    url=url,
    output=output_path_memezip,
    quiet=False
)

### --------------------------------------------------------------------------------------------------------- ###
### --------------------------------------------Unzip meme dataset------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
# Unzip file. via
# https://www.geeksforgeeks.org/unzipping-files-in-python/
with ZipFile(output_path_memezip, "r") as zObject:
    zObject.extractall(path=relative_path)

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------Remove zipped dataset----------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
os.remove(path=output_path_memezip)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------Download annotation dataset-------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
output_path_annotation = os.path.join(
    relative_path,
    "total_def_meme.json"
)
urllib.request.urlretrieve(
    url="https://raw.githubusercontent.com/Social-AI-Studio/Total-Defense-Memes/main/report/annotation.json",
    filename=output_path_annotation
)

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------Load annotation dataset---------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
with open(file=output_path_annotation, mode="r", encoding="utf-8") as jsonfile:
    totaldefmeme_data = json.load(fp=jsonfile)

### --------------------------------------------------------------------------------------------------------- ###
### ---Build dataframe selecting only Singapore-related memes where at least two annotators agree on stance-- ###
### --------------------------------------------------------------------------------------------------------- ###
meme_list = [
    (meme, pillars_and_stances)
    for record in totaldefmeme_data["Pillar_Stances"]
    for meme, pillars_and_stances in record.items()
    if any(len(set(stances)) < len(stances) for _, stances in pillars_and_stances)
]
totaldefmeme_df = pd.DataFrame(
    data=meme_list,
    columns=["image", "pillars_and_stances"]
)

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------Prefix relative directory to file names--------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
totaldefmeme_df.image = totaldefmeme_df.image.astype(str).apply(lambda x: os.path.join(relative_path, "TD_Memes", x))

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------Convert every image to RGB, save as PNG, and update path------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
totaldefmeme_df.image = totaldefmeme_df.image.p_apply(lambda x: (Image.open(x).convert("RGB").save(os.path.splitext(x)[0] + ".png") or os.path.splitext(x)[0] + ".png"))

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------Define a function to upload meme files to S3------------------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(50)
)
def upload_to_s3(
        image_path,
        bucket_name
):
    """
    This function uploads meme files to S3.
    This leverages code from https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/upload_file.html.

            Parameters:
                    image_path (str): File path to meme.
                    bucket_name (str): S3 bucket name.

            Returns:
                    object_url (str): URL to object in S3 bucket.
    """
    s3 = boto3.client("s3")
    s3.upload_file(
        Filename=image_path,
        Bucket=bucket_name,
        Key=Path(image_path).name,
        ExtraArgs={"ContentType": "image/png"}
    )
    object_url = f"https://{bucket_name}.s3.amazonaws.com/{Path(image_path).name}"
    return object_url


### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------Upload meme files to S3--------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
totaldefmeme_df["meme_s3"] = totaldefmeme_df.image.p_apply(
    lambda x: upload_to_s3(
        image_path=x,
        bucket_name=args.bucketname
    )
)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------Add column indicating source of memes---------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
totaldefmeme_df["source"] = "TotalDefMeme"

### --------------------------------------------------------------------------------------------------------- ###
### -----------------------Save dataframe containing meme metadata and uploaded S3 URLs---------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
totaldefmeme_df.to_csv(
    os.path.join(relative_path, "TotalDefMeme.csv"),
    index=False,
    encoding="utf-8"
)
