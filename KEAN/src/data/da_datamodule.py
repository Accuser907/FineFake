from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import pickle as pkl
import torch.nn.functional as F
from transformers import BertTokenizer
import pandas as pd
import os
import re
import fnmatch

class DADataset(Dataset):
    def __init__(self, data_path,num_classes,domain_label,image_path):
        self.data_path = data_path
        #self.transform = transform
        #self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.num_classes = num_classes
        self.domain_label = domain_label
        self.image_path = image_path
        self.samples = self.load_samples()

    def load_samples(self):
        # load_samples here, this is a function to read data
        with open(self.data_path,"rb") as f:
            data_df = pkl.load(f)
        #print(data_df.columns)
        # image_path_list = []
        text_list = list(data_df["text"])
        #image_path_list = ["/data1/zzy/FakeNewsCode/fake-news-baselines/data/SIGIR24/Snopes/snopes_images/0.jpeg" for i in range(len(text_list))]
        image_path_list = list(data_df["image_path"])
        knowledge_list = list(data_df["knowledge_embedding"])
        # for i,knowledge_e in enumerate(knowledge_list):
        #     knowledge_list[i] = torch.tensor(knowledge_e)
        # the following code is to combine image with current path
        for i,image in enumerate(image_path_list):
            image_path_list[i] = os.path.join(self.image_path,image)
        label_list = list(data_df["label"])
        domain_label_list = []
        for i,text in enumerate(text_list):
        #     # if len(text) > 70:
        #     #     text_list[i] = text[:70]
            domain_label_list.append(self.domain_label)
        combined_list = list(zip(text_list,image_path_list,knowledge_list,label_list,domain_label_list))
        # combined_list:[("text",0),("text1",1)]
        return combined_list
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text, image_path, knowledge_e, label, domain_label = self.samples[idx]
        knowledge_e = torch.tensor(knowledge_e).float()
        # data preprocessing
        target = torch.tensor(int(label))
        final_target = F.one_hot(target,num_classes=self.num_classes).float()
        domain_label = torch.tensor(int(domain_label))
        domain_label = F.one_hot(domain_label,num_classes=2).float()
        # if self.tokenizer is not None:
        #     text = self.tokenizer(text,return_tensors='pt',truncation = True,padding='max_length',max_length=512)
        #print(text.shape)
        return text,image_path,knowledge_e,final_target,domain_label
    
    def find_image_path(self,id,origin_path):
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
        
class DADataModule(LightningDataModule):
    def __init__(
        self,
        data_dir_s : str = "data/",
        data_dir_v : str = "data/",
        data_dir_t : str = "data/",
        #train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes : int = 2,
        image_path : str = "",
    ):
        super().__init__()
        #self.data_dir = data_dir
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    @property
    def num_classes(self):
        return 2
    
    def prepare_data(self):
        pass
    
    def setup(self, stage: Optional[str] = None):
        dataset_s = DADataset(data_path = self.hparams.data_dir_s,num_classes=self.hparams.num_classes,domain_label = 0,image_path=self.hparams.image_path)
        dataset_t = DADataset(data_path = self.hparams.data_dir_t,num_classes=self.hparams.num_classes,domain_label = 1,image_path=self.hparams.image_path)
        if stage=="fit" or stage is None:
            # self.data_train_s, self.data_val_s, _ = random_split(
            #     dataset=dataset_s,
            #     lengths=self.hparams.source_split,
            #     generator=torch.Generator().manual_seed(42),
            # )
            # self.data_train_t, self.data_val_t, self.data_test_t = random_split(
            #     dataset=dataset_t,
            #     lengths=self.hparams.target_split,
            #     generator=torch.Generator().manual_seed(42),
            # )
            self.data_train_s = dataset_s
            self.data_train_t = dataset_t
            self.data_val = dataset_t
            self.data_test = dataset_t

    def train_dataloader(self):
        return {"source":DataLoader(
            dataset=self.data_train_s,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        ),
                "target":DataLoader(
            dataset=self.data_train_t,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )}
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
    
if __name__ == "__main__":
    test = DADataModule()