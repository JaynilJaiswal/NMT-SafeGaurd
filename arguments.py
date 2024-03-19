import argparse
import os

def params():
    parser = argparse.ArgumentParser()
    
    # Others
    parser.add_argument("--input-dir", default='assets', type=str, 
                help="The input training data file (a text file).")
    parser.add_argument("--output-dir", default='results', type=str,
                help="Output directory where the model predictions and checkpoints are written.")
    parser.add_argument("--seed", default=42, type=int)
    
    # Hyper-parameters for tuning
    parser.add_argument("--temperature", default=0.07, type=float, 
                help="temperature parameter for contrastive loss")
    parser.add_argument("--batch-size", default=16, type=int,
                help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--learning-rate", default=1e-3, type=float,
                help="Model learning rate starting point.")
    parser.add_argument("--hidden-dim", default=128, type=int,
                help="Model hidden dimension.")
    parser.add_argument("--drop-rate", default=0.1, type=float,
                help="Dropout rate for model training")
    parser.add_argument("--embed-dim", default=768, type=int,
                help="The embedding dimension of pretrained LM.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=10, type=int,
                help="Total number of training epochs to perform.")
    parser.add_argument("--max-len", default=20, type=int,
                help="maximum sequence length to look back")

    args = parser.parse_args()
    return args
