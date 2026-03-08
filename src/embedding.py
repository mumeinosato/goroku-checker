import torch
import torch.nn as nn

class embedding:
    def __init__(self, vocab_size=1024, embedding_dim=128):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    