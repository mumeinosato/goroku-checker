import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels_per_kernel):
        super(CNN, self).__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels_per_kernel, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels, out_channels_per_kernel, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(in_channels, out_channels_per_kernel, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x3 = torch.max(self.relu(self.conv3(x)), dim=2)[0]
        x4 = torch.max(self.relu(self.conv4(x)), dim=2)[0]
        x5 = torch.max(self.relu(self.conv5(x)), dim=2)[0]
        return torch.cat((x3, x4, x5), dim=1)