import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# Define BaseInstance class
class BaseInstance(object):
    def __init__(self, embedding, text, attention_mask):
        self.embedding = embedding  # Input embeddings
        self.text = text  # Original text
        self.attention_mask = attention_mask  # Attention mask

# Custom Dataset class
class WMT14Dataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.data = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data['train'][idx]
        input_ids = self.tokenizer.encode(sentence['translation']['en'], return_tensors="pt", padding=True, truncation=True)
        attention_mask = input_ids!= self.tokenizer.pad_token_id
        return BaseInstance(embedding=input_ids, text=sentence, attention_mask=attention_mask)

    def collate_fn(self, batch):
        # Separate input_ids and attention_masks
        input_ids = [item.embedding for item in batch]
        # attention_masks = [torch.ones_like(input_id) for input_id in input_ids]  # Assuming all tokens are valid
        attention_masks = [item.attention_mask for item in batch]
        # Pad sequences to the maximum length in the batch
        max_len = max(input_id.size()[1] for input_id in input_ids)
        input_ids = torch.stack([torch.nn.functional.pad(input_id, (0, max_len - input_id.size()[1])) for input_id in input_ids])
        attention_masks = torch.stack([torch.nn.functional.pad(mask, (0, max_len - mask.size()[1])) for mask in attention_masks])

        return input_ids, attention_masks
        
    def load_data(self, data_file):
        # Implement loading and preprocessing of data from the file
        pass


def load_model(model, model_name):
    '''Check if pre-trained models exist and load them'''
    try:
        model.load_state_dict(torch.load(model_name))
        print(f"Pre-trained model '{model_name}' loaded successfully.")
        return True
    except FileNotFoundError:
        print(f"No pre-trained model found. Training from scratch.")
        return False

def plot_images(real_images, generated_images, epoch):
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))
    for ax, image in zip(axes[0], real_images):
        ax.imshow(image.view(28, 28).cpu().detach().numpy(), cmap='gray')
        ax.axis('off')
    for ax, image in zip(axes[1], generated_images):
        ax.imshow(image.view(28, 28).cpu().detach().numpy(), cmap='gray')
        ax.axis('off')
    plt.savefig(f'comparison_epoch_{epoch}.png')
    plt.close()