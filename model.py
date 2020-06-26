"""Models:
    ConvNN: DQN
    Dueling DQN: Dueling DQN 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNN(nn.Module):
    """Mainly convNN => Deep Q-Network
    With BatchNorm for each layer
    """
    def __init__(self, in_channels=4, n_actions=6):
        super(ConvNN, self).__init__()
        self.cvlayer = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)
    def forward(self, x):
        x = x.float() / 255.0
        out = self.cvlayer(x)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class DuelingDQN(nn.Module):
    """Dueling version of DQN, seperate the advantage layer and value layer
    """
    def __init__(self, in_channels=4, n_actions=6):
        super(DuelingDQN, self).__init__()
        self.n_actions = n_actions
        self.cvlayer = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.fc1_adv = nn.Linear(64 * 7 * 7, 512)    # advantage value
        self.fc2_adv = nn.Linear(512, n_actions)
        self.fc1_value = nn.Linear(64 * 7 * 7, 512)    # value function
        self.fc2_value = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float() / 255.0   # normaliza of image
        out = self.cvlayer(x)
        out = out.reshape(out.size(0), -1)
        
        adv = F.relu(self.fc1_adv(out))
        value = F.relu(self.fc1_value(out))

        adv = self.fc2_adv(adv)
        value = self.fc2_value(value).expand(x.size(0), self.n_actions) # one-hot action vec
        
        out = value + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.n_actions)
        return out