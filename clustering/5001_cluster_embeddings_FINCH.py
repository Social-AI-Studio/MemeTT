### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import argparse
import ast
import os

import numpy as np
import pandas as pd

from finch import FINCH
from parallel_pandas import ParallelPandas

ParallelPandas.initialize(n_cpu=64)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------Specify required command line arguments-------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description="Cluster embeddings")
parser.add_argument(
    "--inputfile",
    required=True,
    help="Absolute file path of the .csv file containing embeddings"
)
parser.add_argument(
    "--embeddingscol",
    required=True,
    help="Name of embeddings column"
)
parser.add_argument(
    "--outputfile",
    required=True,
    help="Path to the output .csv file containing cluster labels"
)
args = parser.parse_args()


### --------------------------------------------------------------------------------------------------------- ###
### -------------------Create local output directory to store dataframe containing clusters------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
relative_path = os.path.basename(os.path.dirname(args.outputfile))
os.makedirs(
    name=relative_path,
    exist_ok=True
)


### --------------------------------------------------------------------------------------------------------- ###
### -----------------------------------Define a function to cluster embeddings------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
def cluster_embeddings(embeddings):
    """
    This function calls the FINCH algorithm to cluster embeddings and retrieve numeric cluster labels.
    This leverages code from https://github.com/ssarfraz/FINCH-Clustering

            Parameters:
                    embeddings (numpy.ndarray): Input matrix with observations as rows and features as columns.

            Returns:
                    cluster_label_numeric (numpy.ndarray): Array of numeric cluster labels.
    """
    c, num_clust, req_c = FINCH(
        data=embeddings,
        distance="cosine",
        verbose=False
    )
    partition = 0
    cluster_label_numeric = c[:, partition]
    return cluster_label_numeric


### --------------------------------------------------------------------------------------------------------- ###
### --------------------Define main function that applies the cluster_embeddings function-------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
def main():
    memes = pd.read_csv(
        filepath_or_buffer=args.inputfile,
        encoding="utf-8"
    )
    memes[args.embeddingscol] = memes[args.embeddingscol].p_apply(lambda x: ast.literal_eval(x))
    mask = memes[args.embeddingscol].p_apply(lambda emb: len(emb) > 0)
    memes = memes[mask].reset_index(drop=True)
    memes["cluster"] = cluster_embeddings(np.vstack(memes[args.embeddingscol].values))
    memes.to_csv(
        os.path.join(os.path.basename(os.path.dirname(args.outputfile)), os.path.basename(args.outputfile)),
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
