import torch 
import argparse

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--max_length', type=int, default=512)
    
    
    parser.add_argument('--model', type=str, default='transformerEncoder')
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--feedforward_dim', type=int, default=128)
    parser.add_argument('--dropout_rate', type=float, default=0.1)

    
    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--save_itv', type=int, default=125)

    args = parser.parse_args()
    return args
