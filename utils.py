import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np
import os
import pickle

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def setup_gpus(args):
    n_gpu = 0  # set the default to 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu
    if n_gpu > 0:   # this is not an 'else' statement and cannot be combined
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return args

def check_directories(args):
    task_path = os.path.join(args.output_dir)
    if not os.path.exists(task_path):
        os.mkdir(task_path)
        print(f"Created {task_path} directory")
    
    folder = args.task
    
    save_path = os.path.join(task_path, folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f"Created {save_path} directory")
    args.save_dir = save_path

    cache_path = os.path.join(args.input_dir, 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        print(f"Created {cache_path} directory")

    if args.debug:
        args.log_interval /= 10

    return args

# Define BaseInstance class
class BaseInstance(object):
    def __init__(self, embedding, text, attention_mask):
        self.embedding = embedding  # Input embeddings
        self.text = text  # Original text
        self.input_mask = attention_mask  # Attention mask

# Custom Dataset class
class WMT14Dataset(Dataset):
    def __init__(self, file, tokenizer):
        self.data = []
        self.load_data(file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(idx)
        return self.data[idx]
    
    def collate_fn(self, batch):
        input_ids = torch.stack([item.embedding.squeeze() for item in batch])
        attention_masks = torch.stack([item.input_mask.squeeze() for item in batch])
        return input_ids, attention_masks
            
    def load_data(self, data_file):
        # Implement loading and preprocessing of data from the file
        self.data = pickle.load(open(data_file,"rb"))['en']
        