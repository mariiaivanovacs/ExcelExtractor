import pandas as pd 
import os 
import cv2 
import numpy as np 


def sort(path: str):
    df = pd.read_csv(path)
    # sort by nubmers and letters
    df = df.sort_values(by="filename", ascending=True)

    df.to_csv(path, index=False)
    # print(f"Sorted {path}")

import argparse
if __name__ == "__main__":
    # get arguments 
    parser = argparse.ArgumentParser(description="Run image processing pipeline step.")
    parser.add_argument(
        "--file",
        required=True,
        help="Choose which processing step to run."
    )
    args = parser.parse_args()
    
    path = args.file

    
    sort(path)
