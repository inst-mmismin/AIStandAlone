import torch 
import numpy as np 

from torchvision import datasets 
from torchvision.transforms import ToTensor

cifar_mean = [0.49139968, 0.48215827, 0.44653124]
cifar_std = [0.24703233, 0.24348505, 0.26158768]
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

def get_data_mean_and_std(data_type='cifar', data_path='../data'):
    if data_type == 'cifar' : 
        tmp_dataset = datasets.CIFAR10(root=data_path, transform=ToTensor(), train=True, download=True)
    else :
        raise NotImplementedError
    
    mean = tmp_dataset.data.mean(axis=(0, 1, 2)) / 255
    std = tmp_dataset.data.std(axis=(0, 1, 2)) / 255

    return mean.tolist(), std.tolist()


def evaluation(model, test_loader, args):
    model.eval()
    correct = 0 
    total = 0 
    with torch.no_grad(): 
        for idx, (image, label) in enumerate(test_loader): 
            image = image.to(args.device)
            label = label.to(args.device) 

            output = model(image)
            _, pred = torch.max(output, 1)
            correct += (pred == label).sum().item()
            total += image.shape[0]
    model.train()
    return correct / total

def eval_by_class(model, test_loader, args):
    model.eval()
    correct = np.zeros(args.num_classes)
    total = np.zeros(args.num_classes)
    with torch.no_grad(): 
        for idx, (image, label) in enumerate(test_loader): 
            image = image.to(args.device)
            label = label.to(args.device) 

            output = model(image)
            _, pred = torch.max(output, 1)
            for i in range(args.num_classes):
                correct[i] += ((pred == i) * (label == i)).sum().item()
                total[i] += (label == i).sum().item()
    model.train()
    return correct / total, sum(correct) / sum(total)
