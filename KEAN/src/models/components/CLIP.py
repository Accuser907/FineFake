import torch
import clip
from PIL import Image
import os
from torch import fused_moving_avg_obs_fake_quant, nn
from cross_attention import CrossAttention
import sys
sys.path.append("/data1/zzy/FakeNewsCode/fake-news-baselines/src/models/components")
sys.path.append("/data1/zzy/FakeNewsCode/fake-news-baselines")
from src.models.clip_module import ClipModule

class CLIPModel(nn.Module):
    def __init__(self
                 ):
        super().__init__()
        self.model,self.preprocess = clip.load("ViT-B/32",device="cuda")
        for param in self.model.parameters():
            param.requires_grad = False
        self.classifier = classifier(input_size=1184,num_class=2)
        self.sigmoid = nn.Sigmoid()
        self.cross_attention = CrossAttention(input_dim_1=672,input_dim_2=512,hidden_dim = 512).to("cuda")
        
    def forward(self, text,image_path):
        #text,image_path = batch
        text = clip.tokenize(text,context_length=77,truncate=True).to("cuda")
        text_features = self.model.encode_text(text)
        text_features = text_features.squeeze(0)
        if text_features.dim()==1:
            text_features = text_features.unsqueeze(0)
        image_features_list = []
        for image in image_path:
            image_features = self.preprocess(Image.open(image)).unsqueeze(0).to("cuda")
            #image_features = image_features.squeeze(0)
            # image_features = torch.mean(image_features,dim=[3])
            # image_features = image_features.view(1,-1)
            # image_features_list.append(image_features)
            image_features = self.model.encode_image(image_features)
            image_features = image_features.squeeze(0)
            image_features_list.append(image_features)
        # fusion all the image_features to one tensor
        image_features_batch = torch.stack(image_features_list)
        image_features_batch = image_features_batch.squeeze()
        if image_features_batch.dim()==1:
            image_features_batch = image_features_batch.unsqueeze(0)
        # print size
        #print(image_features_batch.size())
        #print(text_features.size())
        # the following will be changed by an attention-fusion model
        fused_features = torch.cat((text_features,image_features_batch),dim=1)
        #outputs = self.classifier(fused_features)
        #logits = self.sigmoid(outputs)
        #fused_features = self.cross_attention(image_features_batch,text_features,text_features)
        return fused_features
    
class classifier(nn.Module):
    def __init__(
        self,
        input_size,
        num_class,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Linear(512,num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        #batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)
        return self.model(x)
    
if __name__ == "__main__":
    text = "test"
    image_path = ["/data1/zzy/FakeNewsCode/fake-news-baselines/data/SIGIR24/Image/twitter/3.jpeg"]
    #data = [text,image_path]
    Net = CLIPModel()
    out = Net(text,image_path)