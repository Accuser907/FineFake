from torch import nn
from transformers import BertModel,BertTokenizer

class BERT(nn.Module):
    def __init__(
        self,
        bert_weighted_path : str,
        tokenizer : str,
    ):
        super().__init__()
        self.text_model = BertModel.from_pretrained(bert_weighted_path)
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.classifier = nn.Linear(768,2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, batch):
        # the following code is a bug because it's not batch size and sequence length but a dic
        batch = self.tokenizer(batch,return_tensors='pt',truncation = True,padding='max_length',max_length=512)
        batch = {key: value.to("cuda") for key, value in batch.items()}
        text_outputs = self.text_model(**batch)
        # input_shape = batch["input_ids"].shape
        # batch_size, seq_length = input_shape[0],input_shape[1]
        text_embedding = text_outputs.last_hidden_state[:, 0, :].detach()
        text_embedding = text_embedding.squeeze()
        #outputs = self.classifier(text_embedding)
        #logits = self.sigmoid(outputs)
        return text_embedding