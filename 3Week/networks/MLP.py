import torch.nn as nn

class myMLP(nn.Module): 
    def __init__(self, hidden_size, num_classes): 
        super().__init__()
        self.hidden_layer1 = nn.Linear(28*28, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        x = self.hidden_layer1(x) 
        x = self.hidden_layer2(x) 
        x = self.hidden_layer3(x) 
        return x