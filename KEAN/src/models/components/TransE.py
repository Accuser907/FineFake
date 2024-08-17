import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
def read_data(data_path):
    """
        data is stored in dataframe architecture
    """
    with open(data_path,"rb") as f:
        data_df = pkl.load(f)
    entity_id = list(data_df["entity_id"])
    return data_df,entity_id
    
def get_embedding(entity_index_file,entity_embedding_file,write_path,data_path):
    """
        get the embedding for each data, and write them in a pickle_file
    """
    with open(entity_index_file,"rb") as f:
        entity_index_dic = pkl.load(f)
    with open(entity_embedding_file,"rb") as f:
        entity_embedding_list = pkl.load(f)
    data_df,entity_id_list = read_data(data_path)
    total_embedding_list = []
    for entity_id in tqdm(entity_id_list):
        embedding_list = []
        for id in entity_id:
            try:
                index = entity_index_dic[id]
                embedding = entity_embedding_list[int(index)]
                embedding_list.append(embedding)
            except Exception as e:
                print(e)
                continue
        average = np.mean(embedding_list,axis=0)
        total_embedding_list.append(average)
    if len(total_embedding_list) == len(data_df):
        print("length is match...")
        data_df["knowledge_embedding"] = total_embedding_list
        with open(write_path,"wb") as f:
            pkl.dump(data_df,f)
        print(data_df.columns)
    else:
        print("Error : length is not match.")
        
if __name__ == "__main__":
    get_embedding(entity_index_file="src/models/components/TransE/entity_index.pkl",entity_embedding_file="src/models/components/TransE/embedding_id.pkl",write_path="data/SIGIR24/Social_Media/add_ke_twitter.pkl",data_path="data/SIGIR24/Social_Media/add_entity_twitter.pkl")