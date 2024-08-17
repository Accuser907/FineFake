import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, hidden_dim):
        super(CrossAttention, self).__init__()
        self.query_linear = nn.Linear(input_dim_1, hidden_dim)
        self.key_linear = nn.Linear(input_dim_2, hidden_dim)
        self.value_linear = nn.Linear(input_dim_2, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.output_linear = nn.Linear(hidden_dim, input_dim_2)
        
    def forward(self, query, key, value):
        query,key,value = query.float(),key.float(),value.float()
        query_proj = self.query_linear(query)
        key_proj = self.key_linear(key)
        value_proj = self.value_linear(value)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query_proj, key_proj.transpose(-2, -1))
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention to value
        attended_value = torch.matmul(attention_weights, value_proj)
        
        # Output
        #output = self.output_linear(attended_value)
        return attended_value

if __name__ == "__main__":
    # Example usage
    input_dim = 128
    hidden_dim = 64
    input_dim_1 = 128
    input_dim_2 = 64
    query = torch.randn(32, input_dim_1).cuda()  # Batch size: 32, Input dimension: 128
    key = torch.randn(32, input_dim_2).cuda()    # Batch size: 32, Input dimension: 128
    value = torch.randn(32, input_dim_2).cuda()  # Batch size: 32, Input dimension: 128
    print(query)
    print(key)
    print(value)
    cross_attention = CrossAttention(input_dim_1, input_dim_2, hidden_dim).to("cuda")
    output = cross_attention(query, key, value)
    print(output.shape)  # Output shape: [32, 128]
