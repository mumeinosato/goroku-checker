import torch.nn as nn
from src.embedding import Embedding
from src.ccn import CNN

class Model(nn.Module):
    def __init__(self, vocab_size=1024, embedding_dim=128):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.cnn = CNN(embedding_dim, 100)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.cnn(x)
        return x