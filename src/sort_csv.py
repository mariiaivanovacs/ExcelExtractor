import pandas as pd 
import os 
import cv2 
import numpy as np 


def sort(path: str):
    df = pd.read_csv(path)
    # sort by nubmers and letters
    df = df.sort_values(by="filename", ascending=True)

    df.to_csv(path, index=False)
    print(f"Sorted {path}")


if __name__ == "__main__":
    sort("data/csv/numbers.csv")
    sort("data/csv/other.csv")