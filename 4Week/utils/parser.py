import torch 
import argparse

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--model', type=str, default='vgg', choices=['LeNet', 'vgg'])
    parser.add_argument('--vgg_type', type=str, default='A', 
                        choices=['A', 'B', 'C', 'D', 'E'])
    
    parser.add_argument('--results_folder', type=str, default='results')

    args = parser.parse_args()
    return args

def infer_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_folder', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    return args

def load_args_from_dict(dict): 
    parser = argparse.Namespace(**dict)
    return parser