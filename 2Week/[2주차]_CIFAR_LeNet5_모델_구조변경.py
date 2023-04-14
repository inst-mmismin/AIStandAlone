"""
# 2주차에서 다루는 기본 학습 코드 

- 목표 
    - 평가 코드 이해하기
    - 이미지 데이터 전처리 이해하기
    - 모델 구현 과정 고도화 
- 주의사항 : 실시간 코딩으로 구체적인 구현체(함수, 변수명 등)의 차이가 있을 수 있음 


- 사용 데이터 : CIFAR10 
- 사용 모델 : LeNet5
"""

# 필요한 패키지 업로드 
import torch 
import torch.nn as nn 
from torch.optim import Adam 
from torchvision import datasets 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import Normalize

import cv2 
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 하이퍼파라메터 세팅 
batch_size = 100 
num_classes = 10 
hidden_size = 500
lr = 0.001
epochs = 5 

image_size = 32

# CIFAR 데이터 관련 (dataset, dataloader)

# 통계량 측정 
tmp_dataset = datasets.CIFAR10(root='./cifar', transform=ToTensor(), train=True, download=True)
mean = tmp_dataset.data.mean(axis=(0, 1, 2)) / 255
std = tmp_dataset.data.std(axis=(0, 1, 2)) / 255

trans = Compose([
    Resize(image_size), 
    ToTensor(),
    Normalize(mean.tolist(), std.tolist())
])

train_dataset = datasets.CIFAR10(root='./cifar', transform=trans, train=True, download=True)
test_dataset = datasets.CIFAR10(root='./cifar', transform=trans, train=False, download=True)

# dataset 이미지 확인하기 (tensor -> numpy) 
def reverse_trans(x):
    x = (x * std) + mean
    return x.clamp(0, 1) * 255 

def get_numpy_image(data): 
    img = reverse_trans(data.permute(1, 2, 0)).type(torch.uint8).numpy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
idx = 1000 

img, label = train_dataset.__getitem__(idx)
img = cv2.resize(
    get_numpy_image(img), 
    (512, 512)
)
label = labels[label]
# cv2.imshow(label, img)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# 모델 class 짜기 
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

# Loss와 Optimizer Class 짜기 
pass

# 모델, Loss, Optimizer 객체 만들기 
model = myLeNet5_int_linear(num_classes).to(device)
model = myLeNet5_conv(num_classes).to(device)
model = myLeNet5_multi_conv(num_classes).to(device)
criteria = nn.CrossEntropyLoss() 
optim = Adam(model.parameters(), lr=lr)

# 평가 코드 
def eval(model, test_loader):
    model.eval()
    correct = 0 
    total = 0 
    with torch.no_grad(): 
        for idx, (image, label) in enumerate(test_loader): 
            image = image.to(device)
            label = label.to(device) 

            output = model(image)
            _, pred = torch.max(output, 1)
            correct += (pred == label).sum().item()
            total += image.shape[0]
    return correct / total

def eval_by_class(model, test_loader):
    model.eval()
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    with torch.no_grad(): 
        for idx, (image, label) in enumerate(test_loader): 
            image = image.to(device)
            label = label.to(device) 

            output = model(image)
            _, pred = torch.max(output, 1)
            for i in range(num_classes):
                correct[i] += ((pred == i) * (label == i)).sum().item()
                total[i] += (label == i).sum().item()
    return correct / total, sum(correct) / sum(total)

# 학습 시각화 
losses = []
accs1 = []
accs2 = []

# 실제 학습 loop 돌리기 
for epoch in range(epochs) : 
    for idx, (image, label) in enumerate(train_loader) : 
        image = image.to(device)
        label = label.to(device) 

        output = model(image)
        loss = criteria(output, label)
        optim.zero_grad()
        loss.backward() 
        optim.step() 
        
        if idx % 100 == 0: 
            print(f'{epoch}/{epochs} , {idx} step | Loss : {loss.item():.4f}')
            losses.append(loss.item())
            accs1.append(eval(model, test_loader))
            accs2.append(eval_by_class(model, test_loader)[1])

import matplotlib.pyplot as plt
plt.plot(losses)
plt.plot(accs1)
plt.plot(accs2)
plt.show()

