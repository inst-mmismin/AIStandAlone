import torch 
import torch.nn as nn 


class myLeNet5(nn.Module): 
    def __init__(self, num_classes): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(16*5*5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x) 
        x = self.relu(x) 
        x = self.pool(x) 
        x = self.conv2(x)
        x = self.bn2(x) 
        x = self.relu(x) 
        x = self.pool(x) 

        x = x.reshape(batch_size, -1)

        x = self.fc1(x)
        x = self.relu(x) 
        x = self.fc2(x)
        x = self.relu(x) 
        x = self.fc3(x)
        return x
