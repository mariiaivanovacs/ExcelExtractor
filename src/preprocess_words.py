import pandas as pd 
import os 
import cv2 
import numpy as np 



DICTONARY_PATH = "data/input/technical_table_terms.csv"

def load_dictionary(path: str) -> pd.DataFrame:
    """Load dictionary from txt file"""
    
    
    import pandas as pd

    # Replace with your actual filenames
    csv1 = "results/features_second_3.csv"
    csv2 = path
    
    df_1 = pd.read_csv(csv1)
    df_2 = pd.read_csv(csv2)
    
    
    # compare columns and its lengths
    print(f"Columns of df_1: {df_1.columns}")
    print(f"Columns of df_2: {df_2.columns}")
    print(f"Len of columns of df_1: {len(df_1.columns)}")
    print(f"Len of columns of df_2: {len(df_2.columns)}")
    
    # output = "results/features_combined.csv"

    # # Read both CSVs
    # df1 = pd.read_csv(csv1)
    # df2 = pd.read_csv(csv2)

    # # Combine them (stack vertically)
    # combined = pd.concat([df1, df2], ignore_index=True)

    # # Save to new CSV
    # combined.to_csv(output, index=False)

    # with open(path, "r", encoding="utf-8") as f:
    #     text = f.read()  # read entire file
    #     words = text.split()  # split by any whitespace
    #     words = [word for word in words if word.strip() and not word.strip()[0].isdigit() ]
    #     words = set(words)

    # print(words)
    # lines = [line.strip() for line in lines if line.strip() and not line.strip()[0].isdigit()]
    # words = [line.split(" ") for line in lines]
    # print(lines[:10])
    # final = []
    # for word in words:
    #     final.extend(word)
    # print("Final:", final)
        
    # replace all , with 
    # words = [word.replace(",", " ") for word in words]
    # print("AA")
    
    # # df = df["term"]
    # # print(df.head(10))
    # # store df in words.csv file
    # # df.to_csv("data/input/technical_table_terms.csv", index=False)
    # with open("data/input/technical_table_terms.csv", "w") as f:
    #     f.writelines("\n".join(words))

    # df = pd.read_csv(
    #     path,
    #     index_col=False,
    #     dtype={"character": str}  # or any type you want
    # )    
    # df = df[["character", "variant", "mass", "center_x", "center_y", "first_moment", "second_moment", "magnitude", "edge_density", "symmetry", "aspect_ratio", "projection_variance", "l1_normalized_gradient"]]
    
    
    # print(df.head(10))
    # save index column (label)
    # df["label"]
    # df.set_index("sample_id")
    # print(df.head(10))
    # df = df[["label", "dark_frac", "aspect_ratio", "compactness", "col_peaks", "vertical_intensity_variance", "intensity_fluctuation_ratio", "frequency_white", "average_peak_width"]]
    # df.to_csv("results/features_second_3.csv", index=False)
    return True

     

def main():
    load_dictionary("experiment/real_features.csv")
    
    # for index, row in df.iterrows():
    
    


if __name__ == "__main__":
    main()