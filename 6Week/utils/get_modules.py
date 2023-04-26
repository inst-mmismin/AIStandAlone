from misc.tools import *

from torchvision import datasets 
from torch.utils.data import DataLoader

from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import Normalize

def get_cifar_transform(args): 
    mean, std = cifar_mean, cifar_std
    trans = Compose([
        Resize((args.image_size, args.image_size)), 
        ToTensor(),
        Normalize(mean, std)
    ])
    return trans 


def get_dataloader(args) : 
    trans = get_cifar_transform(args)
    
    train_dataset = datasets.CIFAR10(root=args.data_path, train=True, transform=trans, download=True)
    test_dataset = datasets.CIFAR10(root=args.data_path, train=False, transform=trans, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader

def get_model(args) : 
    if args.model == 'LeNet' : 
        from networks.LeNet import myLeNet5
        model = myLeNet5(args.num_classes)
    elif args.model == 'vgg' : #  VGG 
        if args.vgg_type == 'A' : 
            from networks.VGG import VGG_A
            model = VGG_A(args.num_classes)
        elif args.vgg_type == 'B' :
            from networks.VGG import VGG_B
            model = VGG_B(args.num_classes)
        elif args.vgg_type == 'C' :
            from networks.VGG import VGG_C
            model = VGG_C(args.num_classes)
        elif args.vgg_type == 'D' :
            from networks.VGG import VGG_D
            model = VGG_D(args.num_classes)
        elif args.vgg_type == 'E' :
            from networks.VGG import VGG_E
            model = VGG_E(args.num_classes)
        else :
            raise NotImplementedError
    elif args.model == 'resnet' : 
        from networks.ResNet import ResNet
        model = ResNet(args.num_classes, args.resnet_config)
        
    else : 
        raise ValueError

    return model