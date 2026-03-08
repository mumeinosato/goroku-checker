import torch.nn as nn
from src.layers import Embedding, CNN, BiLSTM

class Model(nn.Module):
    def __init__(self, vocab_size=1024, embedding_dim=128):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.cnn = CNN(embedding_dim, 100)
        self.bilstm = BiLSTM(input_size=300, hidden_size=128)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.cnn(x)
        x = self.bilstm(x)
        return x