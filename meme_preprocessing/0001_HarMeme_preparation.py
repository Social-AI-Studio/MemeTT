### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
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
parser = argparse.ArgumentParser(description="Prepare HarMeme dataset")
parser.add_argument(
    "--outputdir",
    required=True,
    help="Path to the output folder where the annotation datasets and memes will be saved"
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
### ------------Create local output directory to save the combined TotalDefMeme and HarMeme dataset---------- ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(args.outputdir)
os.makedirs(
    name=relative_path,
    exist_ok=True
)

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------Download HarMeme dataset--------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
# Download large file from Google Drive. via
# https://github.com/wkentaro/gdown
url = "https://drive.google.com/u/0/uc?id=1aMvOHACrG5SgMl4tm9BP1VTiqv8zFn9b"
output_path_memezip = os.path.join(
    relative_path,
    "HarMeme.zip"
)
gdown.download(
    url=url,
    output=output_path_memezip,
    quiet=False
)

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------Unzip HarMeme dataset----------------------------------------- ###
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
### ---------------------------------------Download annotation datasets-------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
# ----------------------------------------------------Harm-P--------------------------------------------------- #
output_path_annotation_harmp_train = os.path.join(
    relative_path,
    "harmp_train.jsonl"
)
urllib.request.urlretrieve(
    url="https://raw.githubusercontent.com/LCS2-IIITD/MOMENTA/main/HarMeme_V1/Annotations/Harm-P/train_v1.jsonl",
    filename=output_path_annotation_harmp_train
)
output_path_annotation_harmp_val = os.path.join(
    relative_path,
    "harmp_val.jsonl"
)
urllib.request.urlretrieve(
    url="https://raw.githubusercontent.com/LCS2-IIITD/MOMENTA/main/HarMeme_V1/Annotations/Harm-P/val_v1.jsonl",
    filename=output_path_annotation_harmp_val
)
output_path_annotation_harmp_test = os.path.join(
    relative_path,
    "harmp_test.jsonl"
)
urllib.request.urlretrieve(
    url="https://raw.githubusercontent.com/LCS2-IIITD/MOMENTA/main/HarMeme_V1/Annotations/Harm-P/test_v1.jsonl",
    filename=output_path_annotation_harmp_test
)
# ----------------------------------------------------Harm-C--------------------------------------------------- #
output_path_annotation_harmc_train = os.path.join(
    relative_path,
    "harmc_train.jsonl"
)
urllib.request.urlretrieve(
    url="https://raw.githubusercontent.com/LCS2-IIITD/MOMENTA/main/HarMeme_V1/Annotations/Harm-C/train.jsonl",
    filename=output_path_annotation_harmc_train
)
output_path_annotation_harmc_val = os.path.join(
    relative_path,
    "harmc_val.jsonl"
)
urllib.request.urlretrieve(
    url="https://raw.githubusercontent.com/LCS2-IIITD/MOMENTA/main/HarMeme_V1/Annotations/Harm-C/val.jsonl",
    filename=output_path_annotation_harmc_val
)
output_path_annotation_harmc_test = os.path.join(
    relative_path,
    "harmc_test.jsonl"
)
urllib.request.urlretrieve(
    url="https://raw.githubusercontent.com/LCS2-IIITD/MOMENTA/main/HarMeme_V1/Annotations/Harm-C/test.jsonl",
    filename=output_path_annotation_harmc_test
)

### --------------------------------------------------------------------------------------------------------- ###
### -----------------------------------------Load annotation datasets---------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
# ----------------------------------------------------Harm-P--------------------------------------------------- #
harmp_train = pd.read_json(
    path_or_buf=output_path_annotation_harmp_train,
    lines=True,
    encoding="utf-8"
)
harmp_val = pd.read_json(
    path_or_buf=output_path_annotation_harmp_val,
    lines=True,
    encoding="utf-8"
)
harmp_test = pd.read_json(
    path_or_buf=output_path_annotation_harmp_test,
    lines=True,
    encoding="utf-8"
)
harmp_train["split"] = "train"
harmp_val["split"] = "val"
harmp_test["split"] = "test"
harmp = pd.concat(
    objs=[harmp_train, harmp_val, harmp_test],
    ignore_index=True
)[["image", "labels", "split"]]
# ----------------------------------------------------Harm-C--------------------------------------------------- #
harmc_train = pd.read_json(
    path_or_buf=output_path_annotation_harmc_train,
    lines=True,
    encoding="utf-8"
)
harmc_val = pd.read_json(
    path_or_buf=output_path_annotation_harmc_val,
    lines=True,
    encoding="utf-8"
)
harmc_test = pd.read_json(
    path_or_buf=output_path_annotation_harmc_test,
    lines=True,
    encoding="utf-8"
)
harmc_train["split"] = "train"
harmc_val["split"] = "val"
harmc_test["split"] = "test"
harmc = pd.concat(
    objs=[harmc_train, harmc_val, harmc_test],
    ignore_index=True
)[["image", "labels", "split"]]

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------Prefix directories to file names of memes------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
# ----------------------------------------------------Harm-P--------------------------------------------------- #
harmp.image = harmp.image.astype(str).apply(lambda x: os.path.join(relative_path, "HarMeme_Images/harmeme_images_us_pol", x))
# ----------------------------------------------------Harm-C--------------------------------------------------- #
harmc.image = harmc.image.astype(str).apply(lambda x: os.path.join(relative_path, "HarMeme_Images/harmeme_images_covid_19", x))

### --------------------------------------------------------------------------------------------------------- ###
### --------------------------Convert all meme images to RGBA mode and save in place------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
harmp.image.p_apply(lambda x: Image.open(x).convert("RGBA").save(x))
harmc.image.p_apply(lambda x: Image.open(x).convert("RGBA").save(x))

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
harmp["meme_s3"] = harmp.image.p_apply(
    lambda x: upload_to_s3(
        image_path=x,
        bucket_name=args.bucketname
    )
)
harmc["meme_s3"] = harmc.image.p_apply(
    lambda x: upload_to_s3(
        image_path=x,
        bucket_name=args.bucketname
    )
)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------Add column indicating source of memes---------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
# ----------------------------------------------------Harm-P--------------------------------------------------- #
harmp["source"] = "Harm-P"
# ----------------------------------------------------Harm-C--------------------------------------------------- #
harmc["source"] = "Harm-C"

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------Save dataframes containing meme metadata and uploaded S3 URLs-------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
# ----------------------------------------------------Harm-P--------------------------------------------------- #
harmp.to_csv(
    path_or_buf=os.path.join(relative_path, "harmp.csv"),
    index=False,
    encoding="utf-8"
)
# ----------------------------------------------------Harm-C--------------------------------------------------- #
harmc.to_csv(
    path_or_buf=os.path.join(relative_path, "harmc.csv"),
    index=False,
    encoding="utf-8"
)
