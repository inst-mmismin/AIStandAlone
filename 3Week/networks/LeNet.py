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

class myLeNet5_int_linear(nn.Module): 
    def __init__(self, num_classes): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.int_linear1 = nn.Linear(6*14*14, 2048)
        self.int_bn = nn.BatchNorm1d(num_features=2048)
        self.int_linear2 = nn.Linear(2048, 6*14*14)
        
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

        x = x.reshape(batch_size, 6*14*14)
        x = self.int_linear1(x)
        x = self.int_bn(x)
        x = self.int_linear2(x)
        x = x.reshape(batch_size, 6, 14, 14)

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

class myLeNet5_conv(nn.Module): 
    def __init__(self, num_classes): 
        super().__init__()
        
        self.convs1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1), 
                nn.BatchNorm2d(num_features=3), 
                nn.ReLU(), 
            ) for _ in range(2) 
        ])
        self.convs1.append(
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1), 
                nn.BatchNorm2d(num_features=6), 
                nn.ReLU(), 
            )
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.convs2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1), 
                nn.BatchNorm2d(num_features=6), 
                nn.ReLU(), 
            ) for _ in range(2) 
        ])
        self.convs2.append(
            nn.Sequential(
                    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), 
                    nn.BatchNorm2d(num_features=16), 
                    nn.ReLU(), 
                )
        )

        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        for m in self.convs1:
            x = m(x)
        x = self.pool(x) 

        for m in self.convs2:
            x = m(x)
        x = self.pool(x)

        x = x.reshape(batch_size, -1)

        x = self.fc(x)
        return x

class myLeNet5_multi_conv(nn.Module): 
    def __init__(self, num_classes): 
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=6, kernel_size=5, stride=1)
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

        x_1 = self.conv1_1(x)
        x_1 = self.bn1(x_1) 
        x_2 = self.conv1_2(x)
        x_2 = self.bn1(x_2) 
        x_3 = self.conv1_3(x)
        x_3 = self.bn1(x_3) 
        x = torch.cat([x_1, x_2, x_3], dim=1)
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
