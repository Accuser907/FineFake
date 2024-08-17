# This code is the instruction of usage of FineFake
# FineFake is stored in dataframe in pickle file, please use the followed commands to use FineFake:
# pip install pickle
# pip install pandas

import pickle as pkl
import pandas as pd

if __name__ == "__main__":
    file_path = "" # the relative path/path of data
    with open(file_path,"rb") as f:
        data_df = pkl.load(f) # data_df is in dataframe
    # print the key columns of data_df
    print(data_df.columns) 
    # if you want extract certain key columns, you can extract it as a list
    # ["news1","news2","news3"...]
    text_list = list(data_df["text"])
    # you can also extract key columns as pandas file
    text_data = data_df["text"]
    # you can extract certain news
    news_data = data_df.iloc[0]
    