import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)
        
    def forward(self, x):
        output, _ = self.attention(x, x, x)
        return output