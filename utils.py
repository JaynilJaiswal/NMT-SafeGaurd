import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np
import os

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
    save_path = os.path.join(args.output_dir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f"Created {save_path} directory")
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f"Created {save_path} directory")
    args.save_dir = save_path

    cache_path = os.path.join(args.input_dir, 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        print(f"Created {cache_path} directory")
    return args

# # Define BaseInstance class
# class BaseInstance(object):
#     def __init__(self, embedding, text, attention_mask):
#         self.embedding = embedding  # Input embeddings
#         self.text = text  # Original text
#         self.attention_mask = attention_mask  # Attention mask

# # Custom Dataset class
# class WMT14Dataset(Dataset):
#     def __init__(self, dataset, tokenizer):
#         self.data = dataset
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sentence = self.data['train'][idx]
#         input_ids = self.tokenizer.encode(sentence['translation']['en'], return_tensors="pt", padding=True, truncation=True)
#         attention_mask = input_ids!= self.tokenizer.pad_token_id
#         return BaseInstance(embedding=input_ids, text=sentence, attention_mask=attention_mask)

#     def collate_fn(self, batch):
#         # Separate input_ids and attention_masks
#         input_ids = [item.embedding for item in batch]
#         # attention_masks = [torch.ones_like(input_id) for input_id in input_ids]  # Assuming all tokens are valid
#         attention_masks = [item.attention_mask for item in batch]
#         # Pad sequences to the maximum length in the batch
#         max_len = max(input_id.size()[1] for input_id in input_ids)
#         input_ids = torch.stack([torch.nn.functional.pad(input_id, (0, max_len - input_id.size()[1])) for input_id in input_ids])
#         attention_masks = torch.stack([torch.nn.functional.pad(mask, (0, max_len - mask.size()[1])) for mask in attention_masks])

#         return input_ids, attention_masks
        
#     def load_data(self, data_file):
#         # Implement loading and preprocessing of data from the file
#         pass