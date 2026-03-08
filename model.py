import torch.nn as nn
from src.layers import Embedding, CNN, BiLSTM, Pooling, Classifier

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, lstm_hidden=128):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.cnn = CNN(embedding_dim, num_filters)
        self.bilstm = BiLSTM(num_filters*3, lstm_hidden)
        self.pooling = Pooling()
        self.classifier = Classifier(lstm_hidden*2)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.cnn(x)
        x = self.bilstm(x)
        x = self.pooling(x)
        x = self.classifier(x)
        return x