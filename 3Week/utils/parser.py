import torch 
import argparse

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--model', type=str, default='LeNet', 
                        choices=['MLP', 'LeNet', 'LeNet_linear', 'LeNet_conv', 'LeNet_multi_conv'])

    args = parser.parse_args()
    return args