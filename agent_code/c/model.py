import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
from collections import namedtuple

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 1000)
        self.fc5 = nn.Linear(1000, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)


# class DQN(nn.Module):
#     def __init__(self, in_channels, n_actions):
#         """
#         Initialize Deep Q Network
#
#         Args:
#             in_channels (int): number of input channels
#             n_actions (int): number of outputs
#         """
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
#         # self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         # self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         # self.bn3 = nn.BatchNorm2d(64)
#         self.fc4 = nn.Linear(7 * 7 * 64, 512)
#         self.head = nn.Linear(512, n_actions)
#
#     def forward(self, x):
#         x = x.float() / 255
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.fc4(x.view(x.size(0), -1)))
#         return self.head(x)