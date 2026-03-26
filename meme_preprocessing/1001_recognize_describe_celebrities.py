### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import os
import requests

import boto3
import pandas as pd

from dotenv import load_dotenv
from io import BytesIO
from parallel_pandas import ParallelPandas
from pathlib import Path
from PIL import Image, ImageDraw
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from wikidata.client import Client

ParallelPandas.initialize(n_cpu=32)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------Specify required command line arguments-------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description="Recognize celebrities and extract information")
parser.add_argument(
    "--inputfile",
    required=True,
    help="Absolute file path of the .csv file containing meme URLs, OCR responses and meme texts for the combined dataset"
)
parser.add_argument(
    "--outputfile",
    required=True,
    help="Path to the output .csv file containing meme URLs, OCR responses and meme texts, recognized celebrities and their information for the combined dataset"
)
parser.add_argument(
    "--outputpath",
    required=True,
    help="Path to the folder where memes with bounding boxes drawn are saved"
)
parser.add_argument(
    "--inputcred",
    required=True,
    help="Absolute file path of the .env file containing AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, AWS_MAX_ATTEMPTS and AWS_RETRY_MODE"
)
parser.add_argument(
    "--minconf",
    required=True,
    help="Minimum MatchConfidence score associated with a celebrity above which information about the celebrity is retrieved; here, celebrity refers to the one with the highest score amongst all celebrities detected for an image",
    type=float)
parser.add_argument(
    "--bucketnameinput",
    required=True,
    help="S3 bucket name where memes to input were saved"
)
parser.add_argument(
    "--bucketnamesave",
    required=True,
    help="S3 bucket name where memes with bounding boxes drawn are saved"
)
args = parser.parse_args()

### --------------------------------------------------------------------------------------------------------- ###
### -------------------Load .env file containing AWS credentials, region and other parameters---------------- ###
### --------------------------------------------------------------------------------------------------------- ###
load_dotenv(Path(args.inputcred))

### --------------------------------------------------------------------------------------------------------- ###
### --------------------------------------Initialize AWS and Wikidata client--------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
wikidata_client = Client()
aws_client = boto3.client("rekognition")

### --------------------------------------------------------------------------------------------------------- ###
### -----------------Create local output directory to store memes with bounding boxes overlaid--------------- ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.join(
    os.path.basename(os.path.dirname(args.outputpath)),
    os.path.basename(args.outputpath)
)
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### --Define a function to call the Amazon Rekognition API to recognize celebrities and retrieve information- ###
### --------------------------------------------------------------------------------------------------------- ###
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(max_attempt_number=20)
)
def identify_and_describe_celebrities(
        image_url,
        bucket_name,
        min_confidence,
        output_dir
):
    """
    This function calls the Amazon Rekognition API to recognize celebrities in memes and retrieve information regarding identified celebrities.
    This leverages code from https://docs.aws.amazon.com/rekognition/latest/dg/celebrities-procedure-image.html
    https://docs.aws.amazon.com/rekognition/latest/dg/get-celebrity-info-procedure.html#get-celebrity-info-examples
    https://github.com/dahlia/wikidata
    For a given meme, the RecognizeCelebrities endpoint is invoked to identify celebrities.
    For up to five identified celebrities with the highest MatchConfidence scores, if the MatchConfidence score equals or exceeds the threshold, the celebrity is recorded and the associated bounding box is drawn on the original meme.
    The ID of the celebrity is also used to retrieve URLs with information about the celebrity.
    If Wikidata is one of the URLs, the entity description is retrieved, if any.

            Parameters:
                    image_url (str): URL of meme.
                    bucket_name (str): S3 bucket name.
                    min_confidence (float): Minimum MatchConfidence score associated with a top five celebrity above which a bounding box is drawn on the meme and information about the celebrity is retrieved.
                    output_dir (str): Path to folder where memes with bounding boxes drawn are saved.

            Returns:
                    tuple containing

                            - recognize_celebrities_response (dict): Response from invoking the RecognizeCelebrities endpoint.
                            - get_celebrity_info_response_list (list | None): List of responses from invoking the GetCelebrityInfo endpoint for the highest-confidence celebrities, if celebrities are detected with a MatchConfidence score equalling or exceeding threshold, else None.
                            - celebrities (list | None): A list of up to five tuples, each comprising the celebrity name, color of associated bounding box drawn on the meme, and the Wikidata description of the celebrity, if celebrities are detected with a MatchConfidence score equalling or exceeding threshold, else None.
                            - rel_to_img_path (str | None): Relative file path to meme with bounding boxes drawn, if celebrities are detected with a MatchConfidence score equalling or exceeding threshold, else None.
    """
    r = requests.get(image_url)
    r.raise_for_status()
    content = r.content
    recognize_celebrities_response = aws_client.recognize_celebrities(Image={"S3Object":{"Bucket": bucket_name, "Name": Path(image_url).name}})
    sorted_top_celebrities = sorted(
        recognize_celebrities_response["CelebrityFaces"],
        key=lambda c: c["MatchConfidence"],
        reverse=True
    )[:5]
    passed_celebrities = [
        c for c in sorted_top_celebrities
        if c["MatchConfidence"] >= min_confidence
    ]
    img = Image.open(BytesIO(content))
    draw = ImageDraw.Draw(img)
    width, height = img.size
    file_name = Path(image_url).name
    celebrities = []
    get_celebrity_info_response_list = []
    box_colors = ["red", "blue", "green", "yellow", "purple"]

    for idx, celebrity in enumerate(passed_celebrities):
        bounding_box = celebrity["Face"]["BoundingBox"]
        left   = bounding_box["Left"] * width
        top    = bounding_box["Top"] * height
        right  = left + (bounding_box["Width"] * width)
        bottom = top  + (bounding_box["Height"] * height)

        color = box_colors[idx]

        draw.rectangle(
            xy=[(left, top), (right, bottom)],
            outline=color,
            width=2
        )
        get_celebrity_info_response = aws_client.get_celebrity_info(Id=celebrity["Id"])
        get_celebrity_info_response_list.append(get_celebrity_info_response)
        wikidata_link = next((link for link in get_celebrity_info_response["Urls"] if "www.wikidata.org/wiki/" in link), None)
        if wikidata_link:
            celebrity_entry = wikidata_client.get(
                Path(wikidata_link).name,
                load=True
            )
            description = str(celebrity_entry.description)
        else:
            description = ""
        celebrities.append((celebrity["Name"], color, description))
    if passed_celebrities:
        rel_to_img_path = os.path.join(
            Path(*Path(output_dir).parts[-2:]).as_posix(),
            file_name
        )
        img.save(rel_to_img_path)
        return recognize_celebrities_response, get_celebrity_info_response_list, celebrities, rel_to_img_path
    else:
        return recognize_celebrities_response, None, None, None

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------Define a function to upload meme files to S3------------------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(max_attempt_number=50)
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
### -------------Define main function that applies the identify_and_describe_celebrities function------------ ###
### --------------------------------------------------------------------------------------------------------- ###
def main():
    memes = pd.read_csv(
        filepath_or_buffer=args.inputfile,
        encoding="utf-8"
    )
    memes[["recognize_celebrities_response", "get_celebrity_info_response_list", "celebrities", "meme_bboxes_drawn"]] = memes.p_apply(
        lambda row: identify_and_describe_celebrities(
            image_url=row["meme_s3"],
            bucket_name=args.bucketnameinput,
            min_confidence=args.minconf,
            output_dir=args.outputpath
        ), 
        axis=1, 
        result_type="expand"
    )
    memes["meme_bboxes_drawn_s3"] = memes.meme_bboxes_drawn.p_apply(
        lambda x: upload_to_s3(
            image_path=x,
            bucket_name=args.bucketnamesave
        ) if x is not None else None
    )
    memes.meme_bboxes_drawn_s3 = memes.meme_bboxes_drawn_s3.fillna(memes["meme_s3"])
    memes.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfile)), os.path.basename(args.outputfile)),
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
