import torch.nn as nn

class VGG_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(VGG_conv, self).__init__()
        padding_size = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding_size)
        self.bn = nn.BatchNorm2d(out_channels) 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class VGG_classifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG_classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

class VGG_A(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG_A, self).__init__()
        self.VGG_block_1 = nn.Sequential(
            VGG_conv(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.VGG_block_2 = nn.Sequential(
            VGG_conv(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.VGG_block_3 = nn.Sequential(
            VGG_conv(128, 256),
            VGG_conv(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.VGG_block_4 = nn.Sequential(
            VGG_conv(256, 512),
            VGG_conv(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.VGG_block_5 = nn.Sequential(
            VGG_conv(512, 512),
            VGG_conv(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = VGG_classifier(num_classes)

    def forward(self, x):
        x = self.VGG_block_1(x)
        x = self.VGG_block_2(x)
        x = self.VGG_block_3(x)
        x = self.VGG_block_4(x)
        x = self.VGG_block_5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG_B(VGG_A):
    def __init__(self, num_classes=1000):
        super(VGG_B, self).__init__(num_classes)
        self.VGG_block_1 = nn.Sequential(
            VGG_conv(3, 64),
            VGG_conv(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.VGG_block_2 = nn.Sequential(
            VGG_conv(64, 128),
            VGG_conv(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

class VGG_C(VGG_B):
    def __init__(self, num_classes=1000):
        super(VGG_C, self).__init__(num_classes)
        self.VGG_block_3 = nn.Sequential(
            VGG_conv(128, 256),
            VGG_conv(256, 256),
            VGG_conv(256, 256, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.VGG_block_4 = nn.Sequential(
            VGG_conv(128, 256),
            VGG_conv(256, 256),
            VGG_conv(256, 256, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.VGG_block_5 = nn.Sequential(
            VGG_conv(512, 512),
            VGG_conv(512, 512),
            VGG_conv(512, 512, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

class VGG_D(VGG_C):
    def __init__(self, num_classes=1000):
        super(VGG_D, self).__init__(num_classes)
        self.VGG_block_3 = nn.Sequential(
            VGG_conv(128, 256),
            VGG_conv(256, 256),
            VGG_conv(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.VGG_block_4 = nn.Sequential(
            VGG_conv(256, 512),
            VGG_conv(512, 512),
            VGG_conv(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.VGG_block_5 = nn.Sequential(
            VGG_conv(512, 512),
            VGG_conv(512, 512),
            VGG_conv(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

class VGG_E(VGG_D):
    def __init__(self, num_classes=1000):
        super(VGG_E, self).__init__(num_classes)
        self.VGG_block_3 = nn.Sequential(
            VGG_conv(128, 256),
            VGG_conv(256, 256),
            VGG_conv(256, 256),
            VGG_conv(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.VGG_block_4 = nn.Sequential(
            VGG_conv(256, 512),
            VGG_conv(512, 512),
            VGG_conv(512, 512),
            VGG_conv(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.VGG_block_5 = nn.Sequential(
            VGG_conv(512, 512),
            VGG_conv(512, 512),
            VGG_conv(512, 512),
            VGG_conv(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
