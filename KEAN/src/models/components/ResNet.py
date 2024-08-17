from torch import embedding, nn
from transformers import BertModel,BertTokenizer
import torchvision.transforms as transforms
from PIL import Image
import io
import torch
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(
        self,
        resnet_weight
    ):
        super().__init__()
        resnet_model = models.resnet18()

        # 加载之前保存的权重
        checkpoint = torch.load(resnet_weight)
        # 逐层复制权重
        for name, param in resnet_model.named_parameters():
            if name in checkpoint:
                param.data = checkpoint[name]
        resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
        # 设置模型为评估模式（不使用 dropout 等）
        resnet_model.eval()
        for param in self.resnet_model.parameters():
            param.requires_grad = False
        self.resnet_model = resnet_model
        
    def forward(self, image_path_list):
        image_list = self.transform_image(image_path_list)
        embedding_list = []
        for image in image_list:
            image_embedding = self.resnet_model(image).squeeze()
            embedding_list.append(image_embedding)
        image_features_batch = torch.stack(embedding_list).squeeze()
        if image_features_batch.dim()==1:
            image_features_batch = image_features_batch.unsqueeze(0)
        return image_features_batch
    
    def transform_image(self,image_path_list):
        """
            image: binary file
            return : image_tensor_list
        """
        image_tensor_list = []
        for image_path in image_path_list:
            # image_data = io.BytesIO(image)
            image = Image.open(image_path)
            # check the channel-num
            if image.mode != "RGB":
                image = image.convert("RGB")
            transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
            #image_tensor = transform(image).unsqueeze(0)
            image_tensor = transform(image)
        image_tensor_list.append(image_tensor)
        return image_tensor_list
    