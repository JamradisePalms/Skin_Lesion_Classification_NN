import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.lin1 = nn.Linear(64 * 6 * 6, 120)
        self.lin2 = nn.Linear(120, 50)
        self.lin3 = nn.Linear(50, 20)
        self.lin4 = nn.Linear(20, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = torch.reshape(x, (128, 64 * 6 * 6))
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return x
