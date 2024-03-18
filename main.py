import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from model import Autoencoder
# from utils import WMT14Dataset
from datasets import load_dataset
from tqdm import tqdm as progress_bar
import torch.nn.functional as F
from utils import set_seed, setup_gpus, check_directories
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params



# Load data from pickle file
# with open('all_inputs_250', 'rb') as f:
#     data = pickle.load(f)
# Load WMT14 dataset from Hugging Face
# dataset = load_dataset("wmt14", "de-en")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class SequenceReconstructionLoss(nn.Module):
    def __init__(self, vocab_size):
        super(SequenceReconstructionLoss, self).__init__()
        self.vocab_size = vocab_size

    def forward(self, pred_sequence, target_sequence):
        """
        Compute sequence-to-sequence reconstruction loss.

        Args:
        - pred_sequence (Tensor): Predicted sequence (shape: [batch_size, sequence_length]).
        - target_sequence (Tensor): Target sequence (ground truth) (shape: [batch_size, sequence_length]).

        Returns:
        - loss (Tensor): Sequence-to-sequence reconstruction loss.
        """
        
        # One-hot encode the predicted and target sequences
        pred_sequence_onehot = F.one_hot(pred_sequence, num_classes=self.vocab_size).float()
        pred_sequence_onehot.requires_grad = True
        # print(pred_sequence_onehot.size())
        # Flatten the predicted sequence to shape [batch_size * sequence_length, vocab_size]
        # pred_sequence_flat = pred_sequence_onehot.view(-1, pred_sequence_onehot.size(-1))

        # Flatten the target sequence to shape [batch_size * sequence_length]
        # target_sequence_flat = target_sequence.view(-1)
        # print(pred_sequence_flat.size(), target_sequence_flat.size())
        # Compute cross-entropy loss
        loss = F.cross_entropy(pred_sequence_onehot.permute(0,2,1), target_sequence)  # Ignore padding index

        return loss

# Define training function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    # print(len(dataloader))
    pbar = progress_bar(enumerate(dataloader), total=len(dataloader))
    
    for step, batch in pbar:        
        # print(batch[1].size())
        inputs, labels = prepare_inputs(batch, use_text=False)
        # print(inputs)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # input_ids = batch[0].squeeze().to(device)
        # attention_mask = input_ids != tokenizer.pad_token_id
        # attention_mask = batch[1].squeeze().to(device)
        optimizer.zero_grad()
        reconstructed_ids = model(input_ids, attention_mask)
        # print(reconstructed_logits.view(-1, reconstructed_logits.size(-1))[0], input_ids.size())
        loss = criterion(reconstructed_ids*attention_mask, input_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"Batch Loss": loss.item(),"Total Loss": total_loss/len(dataloader)})
    return total_loss / len(dataloader)


if __name__ == '__main__':
    
    args = params()
    args = setup_gpus(args)
    args = check_directories(args)
    set_seed(args)
    print(args)
    cache_results, already_exist = check_cache(args)    
    tokenizer = load_tokenizer(args)
    
    if already_exist:
        features = cache_results
    else:
        data = load_data()
        features = prepare_features(args, data, tokenizer, cache_results)
    datasets = process_data(args, features, tokenizer)
    for k,v in datasets.items():
        print(k, len(v))
    # for i in range(10):
    #     print(datasets['test'][i].text, datasets['test'][i].label_text)
    # Initialize dataset and dataloader
    # wmt14_dataset = WMT14Dataset(dataset, tokenizer=tokenizer)
    # dataloader = DataLoader(wmt14_dataset, batch_size=16, shuffle=True, collate_fn=wmt14_dataset.collate_fn)
    dataloader = get_dataloader(args, datasets['train'], split='train')  #/////////// might need to change

    # Initialize model, optimizer, and criterion
    autoencoder = Autoencoder()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    # # criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion = SequenceReconstructionLoss(len(tokenizer))

    # Define device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    autoencoder.to(device)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train(autoencoder, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")