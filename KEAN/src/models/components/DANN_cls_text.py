import torch
import torch.nn as nn
from torch.autograd import Function
import sys
sys.path.append("/data1/zzy/FakeNewsCode/fake-news-baselines/src/models/components")
from CLIP import CLIPModel

class DANN(nn.Module):
    def __init__(
        self,
        #knowledge_size,
        domain_size,
        hidden_size,
        hidden_size_2,
        num_classes,        
    ):
        super().__init__()
        # both image and text
        self.encoder1 = CLIPModel()
        self.encoder_content = Content_Encoder(input_size=1024,output_size=128)
        # the following encoder and decoder is for knowledge
        #self.encoder2 = Knowledge_Encoder(input_size=100,output_size=50)
        #self.decoder1 = Knowledge_Decoder(input_size=50,hidden_size=512,output_size=100)
        #self.discriminator = domain_discriminator(input_size=domain_size,hidden_size=hidden_size,hidden_size_2=hidden_size_2,num_class=2)
        self.classifer = news_classifier(input_size=128,hidden_size=hidden_size,hidden_size_2=hidden_size_2,num_class=num_classes)
        #self.reverselayer = ReverseLayerF()
        
    def forward(self,text,image_path,knowledge,domain_label):
        #text,image_path,domain_label = batch
        fused_features = self.encoder1(text,image_path).float()
        fused_features = self.encoder_content(fused_features).float()
        #knowledge  = knowledge.float()
        #print(knowledge.dtype)
        #knowledge_features = self.encoder2(knowledge)
        #content_feature = torch.concat((knowledge_features,fused_features),dim=1)
        #reconstructed = self.decoder1(knowledge_features)
        #reverse_feature = ReverseLayerF.apply(content_feature,1)
        #discriminator_logits = self.discriminator(reverse_feature)
        classification_logits = self.classifer(fused_features)
        return classification_logits
    
class domain_discriminator(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        hidden_size_2,
        num_class,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_2,num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        #batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)
        return self.model(x)

class news_classifier(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        hidden_size_2,
        num_class,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        #batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)
        return self.model(x)
    
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    
class Knowledge_Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size,output_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        #batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)
        return self.model(x)

class Content_Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        #batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)
        return self.model(x)
    
class Knowledge_Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size,output_size)
        )

    def forward(self, x):
        #batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)
        return self.model(x)