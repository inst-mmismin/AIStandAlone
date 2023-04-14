from misc.tools import *

from torchvision import datasets 
from torch.utils.data import DataLoader

from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import Normalize

def get_dataloader(args) : 
    mean, std = cifar_mean, cifar_std
    trans = Compose([
        Resize((args.image_size, args.image_size)), 
        ToTensor(),
        Normalize(mean, std)
    ])
    
    train_dataset = datasets.CIFAR10(root=args.data_path, train=True, transform=trans, download=True)
    test_dataset = datasets.CIFAR10(root=args.data_path, train=False, transform=trans, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader

def get_model(args) : 
    if args.model == 'MLP' : 
        from networks.MLP import myMLP
        model = myMLP(args.hidden_size, args.num_classes)
    elif args.model == 'LeNet' : 
        from networks.LeNet import myLeNet5
        model = myLeNet5(args.num_classes)
    elif args.model == 'LeNet_linear' : 
        from networks.LeNet import myLeNet5_int_linear
        model = myLeNet5_int_linear(args.num_classes)
    elif args.model == 'LeNet_conv' : 
        from networks.LeNet import myLeNet5_conv
        model = myLeNet5_conv(args.num_classes)
    elif args.model == 'LeNet_multi_conv' : 
        from networks.LeNet import myLeNet5_multi_conv
        model = myLeNet5_multi_conv(args.num_classes)
    else : 
        raise NotImplementedError

    return model