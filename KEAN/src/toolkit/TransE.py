import numpy as np
import pickle as pkl
from tqdm import tqdm
def reshape_np(numpy_array):
    reshaped_array = numpy_array.reshape((int(len(numpy_array)/100),100))
    print(reshaped_array)
    result = [reshaped_array[i,:] for i in range(reshaped_array.shape[0])]
    print(result[:2]) 
    return result

def read_txt(file_path,write_path):
    # read txt and write the entity_id with numpy
    ans_dic = {}
    with open(file_path,"r") as f:
        for i,data in enumerate(f):
            if (i==0):
                continue
            #print(data)
            data = data.split()
            #print(data)
            entity_id,index = data[0],data[1]
            #print(entity_id,index)
            ans_dic[entity_id] = index
    write_pickle(write_path,ans_dic)
    
def write_pickle(write_path,data):
    with open(write_path,"wb") as f:
        pkl.dump(data,f)
        
if __name__ == "__main__":
    vec = np.memmap("src/models/components/TransE/entity2vec.bin",dtype="float32",mode="r")
    vec = np.array(vec)
    print(len(vec))
    print(vec[:5])
    print(vec[100:105])
    result = reshape_np(vec)
    #txt_path = "src/models/components/TransE/entity2id.txt"
    #write_path = "src/models/components/TransE/entity_index.pkl"
    #read_txt(txt_path,write_path)
    write_pickle(write_path="src/models/components/TransE/embedding_id.pkl",data = result)