import os 
from misc.tools import *

from torchvision import datasets 
from torch.utils.data import DataLoader

from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import Normalize

def get_transform(args): 
    if args.model == 'pretrained' : 
        from torchvision.transforms._presets import ImageClassification
        trans = ImageClassification(crop_size=args.image_size, resize_size=args.image_size)
    else : 
        if args.target_image == 'cifar' : 
            mean, std = cifar_mean, cifar_std
        else : # 'dog'
            mean, std = imagenet_mean, imagenet_std

        trans = Compose([
            Resize((args.image_size, args.image_size)), 
            ToTensor(),
            Normalize(mean, std)
        ])
        
    return trans 

def get_dataset(args, trans): 
    if args.target_image == 'cifar': 
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, transform=trans, download=True)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, transform=trans, download=True)
    else : # 'dog'
        if args.dataset_type == 'image_folder': 
            from torchvision.datasets import ImageFolder
            train_dataset = ImageFolder(root=os.path.join(args.data_path, 'train'), transform=trans)
            test_dataset = ImageFolder(root=os.path.join(args.data_path, 'test'), transform=trans)
        
        elif args.dataset_type == 'customTT':
            from modules.dataset import Dog_Dataset
            train_dataset = Dog_Dataset(data_path=os.path.join(args.data_path, 'train'), transform=trans)
            test_dataset = Dog_Dataset(data_path=os.path.join(args.data_path, 'test'), transform=trans)
        
        elif args.dataset_type == 'sklearn':
            from modules.dataset import Dog_Dataset
            from sklearn.model_selection import train_test_split
            tmp_dataset = Dog_Dataset(data_path=args.data_path, transform=trans)
            train_dataset, test_dataset = train_test_split(tmp_dataset, test_size=0.2, random_state=42)

    return train_dataset, test_dataset

def get_dataloader(args) : 
    trans = get_transform(args)
    
    train_dataset, test_dataset = get_dataset(args, trans)

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

    elif args.model == 'pretrained' : 
        from torchvision.models import resnet18, ResNet18_Weights
        import torch.nn as nn 
        weight= ResNet18_Weights.DEFAULT
        model = resnet18(weights=weight)
        
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, args.num_classes)
        
    else : 
        raise ValueError

    return model