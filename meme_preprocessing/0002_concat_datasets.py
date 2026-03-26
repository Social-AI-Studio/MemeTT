### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import os

import pandas as pd

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------Specify required command line arguments-------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description="Concatenate TotalDefMeme and HarMeme datasets")
parser.add_argument(
    "--inputtdef",
    required=True,
    help="Absolute file path of TotalDefMeme.csv"
)
parser.add_argument(
    "--inputharmp",
    required=True,
    help="Absolute file path of harmp.csv"
)
parser.add_argument(
    "--inputharmc",
    required=True,
    help="Absolute file path of harmc.csv"
)
parser.add_argument(
    "--outputfile",
    required=True,
    help="Path to the output .csv file containing the combined TotalDefMeme, Harm-P and Harm-C dataset"
)
args = parser.parse_args()

### --------------------------------------------------------------------------------------------------------- ###
### -----------------Create directory to save TotalDefMeme, Harm-P and Harm-C combined dataset--------------- ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)

### --------------------------------------------------------------------------------------------------------- ###
### -----------------------------------------------Load datasets--------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
totaldefmeme = pd.read_csv(
    filepath_or_buffer=args.inputtdef,
    encoding="utf-8"
)
harmp = pd.read_csv(
    filepath_or_buffer=args.inputharmp,
    encoding="utf-8"
)
harmc = pd.read_csv(
    filepath_or_buffer=args.inputharmc,
    encoding="utf-8"
)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------Stack datasets--------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
memes = pd.concat(
    objs=[totaldefmeme, harmp, harmc],
    ignore_index=True,
    join="outer"
)

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------Save stacked dataset------------------------------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
memes.to_csv(
    os.path.join(relative_path, os.path.basename(args.outputfile)),
    index=False,
    encoding="utf-8"
)
