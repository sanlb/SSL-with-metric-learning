
import torch.nn as nn
import torch.nn.functional as F
from utils.fun_utils import GaussianNoise



class CNN(nn.Module):
    def __init__(self, num_classes, fm1=64, fm2=128):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, fm1, kernel_size=5)
        self.conv2 = nn.Conv2d(fm1, fm2, kernel_size=3)
        self.conv3 = nn.Conv2d(fm2, fm2, kernel_size=3)
        self.fc1 = nn.Linear(fm2, num_classes)

    def forward(self, x):



        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, 128)
        x = self.fc1(x)
        return x