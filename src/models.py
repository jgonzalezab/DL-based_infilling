import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class FeedForwardNN(nn.Module):

    def __init__(self, input_size, dropout_rate=0.25):
        super(FeedForwardNN, self).__init__()

        self.fc1 = nn.Linear(input_size, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)

        self.fc5 = nn.Linear(128, 1)
    
    def forward(self, x):

        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        out = self.fc4(out)
        out = self.dropout4(out)
        out = self.fc5(out)

        return out

class DeepFeedForwardNN(nn.Module):

    def __init__(self, input_size):
        super(DeepFeedForwardNN, self).__init__()

        self.fc1 = nn.Linear(input_size, 2048)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(1024, 512)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(256, 128)
        self.relu5 = nn.ReLU()

        self.fc6 = nn.Linear(128, 64)
        self.relu6 = nn.ReLU()

        self.fc7 = nn.Linear(64, 32)
        self.relu7 = nn.ReLU()

        self.fc8 = nn.Linear(32, 16)
        self.relu8 = nn.ReLU()

        self.fc9 = nn.Linear(16, 1)
    
    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        x = self.relu4(x)

        x = self.fc5(x)
        x = self.relu5(x)

        x = self.fc6(x)
        x = self.relu6(x)

        x = self.fc7(x)
        x = self.relu7(x)

        x = self.fc8(x)
        x = self.relu8(x)

        out = self.fc9(x)

        return out