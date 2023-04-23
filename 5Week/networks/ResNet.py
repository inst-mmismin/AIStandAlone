import torch.nn as nn 

RES18_LAYER = [2, 2, 2, 2]
RES34_LAYER = [3, 4, 6, 3]
RES50_LAYER = [3, 4, 6, 3]
RES101_LAYER = [3, 4, 23, 3]
RES152_LAYER = [3, 8, 36, 3]

RES33_INOUT = [[64, 64], [64, 128], [128, 256], [256, 512]]
RES131_INOUT = [[64, 256], [256, 512], [512, 1024], [1024, 2048]]

class ResNet_front(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x): 
        x = self.conv(x)
        x = self.pool(x)
        return x 
    
class ResNet_end(nn.Module):
    def __init__(self, num_classes=10, config='18'):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        fc_input = 512 if config in ['18', '34'] else 2048 # 33 구조 or 131 구조
        self.fc = nn.Linear(fc_input, num_classes)

    def forward(self, x): 
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x 

class ResNet_block(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=False):
        super().__init__()
        self.downsample = downsample 
        stride = 1 
        if self.downsample : 
            stride = 2
            self.skip_conn = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                          kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1), 
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(), 
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(), 
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        skip_x = x.clone()

        if self.downsample : 
            skip_x = self.skip_conn(skip_x)

        x = self.first_conv(x) 
        x = self.second_conv(x) 

        x = x + skip_x 
        x = self.relu(x)
        return x 
    
class ResNet_bottleneck(nn.Module): 
    def __init__(self, in_channel, out_channel, downsample=False): 
        super().__init__()
        self.downsample = downsample 
        stride = 2 if self.downsample else 1

        self.skip_conn = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                        kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        mid_channel= out_channel // 4 
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, 
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=mid_channel),
            nn.ReLU()
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, 
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(num_features=mid_channel),
            nn.ReLU()
        )
        self.third_conv = nn.Sequential(
            nn.Conv2d(in_channels=mid_channel, out_channels=out_channel, 
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        skip_x = x.clone()
        skip_x = self.skip_conn(skip_x)

        x = self.first_conv(x) 
        x = self.second_conv(x) 
        x = self.third_conv(x) 

        x = x + skip_x 
        return x 
    

class ResNet_middle(nn.Module): 
    def __init__(self, config): 
        super().__init__()
        num_layers, inout = self.get_num_layers_inout(config)
        
        self.layer1 = self.make_layer(inout[0][0], inout[0][1], num_layers[0])
        self.layer2 = self.make_layer(inout[1][0], inout[1][1], num_layers[1], downsample=True)
        self.layer3 = self.make_layer(inout[2][0], inout[2][1], num_layers[2], downsample=True)
        self.layer4 = self.make_layer(inout[3][0], inout[3][1], num_layers[3], downsample=True)
    
    def get_num_layers_inout(self, config) : 
        if config == '18': 
            self.target_block = ResNet_block
            return RES18_LAYER, RES33_INOUT
        elif config == '34' : 
            self.target_block = ResNet_block
            return RES34_LAYER, RES33_INOUT
        elif config == '50' : 
            self.target_block = ResNet_bottleneck
            return RES50_LAYER, RES131_INOUT
        elif config == '101' : 
            self.target_block = ResNet_bottleneck
            return RES101_LAYER, RES131_INOUT
        elif config == '152' : 
            self.target_block = ResNet_bottleneck
            return RES152_LAYER, RES131_INOUT
        
    def make_layer(self, in_channel, out_channel, num_block, downsample=False):
        layer = [self.target_block(in_channel, out_channel, downsample)]
        for _ in range(num_block-1): 
            layer.append(self.target_block(out_channel, out_channel))
        return nn.Sequential(*layer)
    
    def forward(self, x): 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
class ResNet(nn.Module): 
    def __init__(self, num_classes, config): 
        super().__init__()
        self.front = ResNet_front()
        self.middle = ResNet_middle(config)
        self.end = ResNet_end(num_classes, config=config)
    
    def forward(self, x): 
        x = self.front(x)
        x = self.middle(x) 
        x = self.end(x) 
        return x 
