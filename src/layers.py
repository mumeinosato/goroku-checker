import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size=1024, embedding_dim=128):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)
    
    
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels_per_kernel):
        super(CNN, self).__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels_per_kernel, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels, out_channels_per_kernel, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(in_channels, out_channels_per_kernel, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x3 = self.relu(self.conv3(x))
        x4 = self.relu(self.conv4(x))
        x5 = self.relu(self.conv5(x))
        x = torch.cat((x3, x4, x5), dim=1)
        x = x.transpose(1, 2)
        return x
    
    
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        output, _ = self.lstm(x)
        return output
    

class Pooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.max(x, dim=1)[0]
    

class Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return self.sigmoid(x)