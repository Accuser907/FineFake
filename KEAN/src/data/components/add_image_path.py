import pandas as pd
import pickle as pkl
import os
import re
import fnmatch
from tqdm import tqdm

def find_image_path(id,origin_path):
    """
        find image path
    """
    files_list = os.listdir(origin_path)
    pattern = f"{id}.*"
    matching_files = fnmatch.filter(files_list,pattern)
    
    if matching_files:
        image_path = os.path.join(origin_path,matching_files[0])
        return image_path
    else:
        return None
    
if __name__ == "__main__":
    with open("/data1/zzy/FakeNewsCode/fake-news-baselines/data/SIGIR24/Social_Media/add_ke_reddit_label.pkl","rb") as f:
        data_df = pkl.load(f)
    # the following code is to find the image_path of data by matching id with image_list name
    print(data_df.columns)
    # rating = list(data_df["rating"])
    # print(rating[:5])
    origin_path = "/data1/zzy/FakeNewsCode/fake-news-baselines/data/SIGIR24/Social_Media/reddit_image"
    id_list = list(data_df["id"])
    print(id_list[-10:])
    image_path_list = []
    for i,id in tqdm(enumerate(id_list)):
        image_path = find_image_path(id,origin_path)
        if image_path == None:
            data_df.drop(i,inplace=True)
        else:
            image_path_list.append(image_path)
    print(len(data_df))
    print(len(image_path_list))
    data_df["image_path"] = image_path_list
    print(data_df.columns)
    print(list(data_df["id"][-10:]))
    print(list(data_df["image_path"])[-10:])
    # the following code is to delete data with None image_path
    print(len(data_df))
    for i,image_path in enumerate(image_path_list):
        if image_path == None:
            print(i)
            print(data_df.iloc[i])
            break
    #data_df = data_df[data_df["image_path"]!=None]
    data_df = data_df.dropna(subset=["image_path"])
    print(len(data_df))
    with open("/data1/zzy/FakeNewsCode/fake-news-baselines/data/SIGIR24/Social_Media/reddit.pkl","wb") as f:
        pkl.dump(data_df,f)