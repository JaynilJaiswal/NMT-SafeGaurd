import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from model import Autoencoder, IntentModel
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

def load_model(model, model_name):
    try:
        model.load_state_dict(torch.load(model_name))
        print(f"Pre-trained model '{model_name}' loaded successfully.")
        return True
    except FileNotFoundError:
        print(f"No pre-trained model found. Training from scratch.")
        return False
        
class SequenceReconstructionLoss(nn.Module):
    def __init__(self, vocab_size):
        super(SequenceReconstructionLoss, self).__init__()
        self.vocab_size = vocab_size

    def forward(self, pred_logits, target_sequence):
        """
        Compute sequence-to-sequence reconstruction loss.

        Args:
        - pred_sequence (Tensor): Predicted sequence (shape: [batch_size, sequence_length]).
        - target_sequence (Tensor): Target sequence (ground truth) (shape: [batch_size, sequence_length]).

        Returns:
        - loss (Tensor): Sequence-to-sequence reconstruction loss.
        """
        loss = F.cross_entropy(pred_logits.permute(0, 2, 1), target_sequence)
        return loss

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, pred, target):
        batch_size = pred.size(0)
        num_classes = pred.size(1)
        
        # Create a mask to zero out the log probabilities for the target classes
        mask = torch.ones(batch_size, num_classes, dtype=torch.bool, device=pred.device)
        mask[torch.arange(batch_size), target] = 0
        
        # Zero out the log probabilities for the target classes
        pred_masked = pred.masked_fill(mask.unsqueeze(-1), -float('inf'))
        
        # Calculate the loss for each sample
        loss_per_sample = -torch.sum(pred_masked, dim=1) / num_classes
        
        # Calculate the average loss over the batch
        loss = torch.mean(loss_per_sample)
        
        return loss


def run_eval(args, model, datasets, tokenizer, split='validation'):

    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)
    acc = 0
    losses = 0 
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch)
        logits = model(inputs)
        if split == 'test':
            criterian = nn.CrossEntropyLoss()
            loss = criterian(logits, labels)
            losses += loss.item()

        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()

    if split == 'test':
        print(f'{split} acc:', acc/len(datasets[split]), f'| {losses} loss ', f'|dataset split {split} size:', len(datasets[split]))
    else:
        print(f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split]))
    return acc/len(datasets[split])

def baseline_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    # train_dataloader = get_dataloader(...)
    train_dataloader = get_dataloader(args, datasets['train'], split='train')  #/////////// might need to change
    n_batches = len(train_dataloader)

    # task2: setup model's optimizer_scheduler if you have
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*args.n_epochs), num_training_steps=args.n_epochs)
    
    # task3: write a training loop
    train_acc = []
    val_acc = []
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):  # should look like: progress_bar(enumerate(dataloader), total=len(dataloader)) ?
            inputs, labels = prepare_inputs(batch, use_text=False)
            logits = model(inputs) # params: self, args ?
            loss = criterion(logits, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            model.zero_grad()

            losses += loss.item()
        
        scheduler.step()  # Update learning rate schedule
        train_ =  run_eval(args=args, model= model , datasets=datasets, tokenizer= tokenizer, split='train')
        val_ = run_eval(args=args, model= model , datasets=datasets, tokenizer= tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses)

        train_acc.append(train_)
        val_acc.append(val_)
    torch.save(model.state_dict(), "text_classifier.pth") # save model
    
# Define training function
def train(model, intent_classifier, dataloader, optimizer, criterion, adversarial_loss, device):
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
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        
        # Extract token IDs from logits
        decoded_sequences = torch.argmax(logits, dim=-1)  # Shape: (batch_size, sequence_length)
        
        # Pass token IDs to the intent classifier
        intent_logits = intent_classifier(decoded_sequences*attention_mask)
        
        reconstruction_loss = criterion(logits, input_ids)
        
        # Adversarial loss
        with torch.no_grad():
            adversarial_target = torch.argmax(intent_logits, dim=1)  # get the target class
        adversarial_loss_value = adversarial_loss(intent_logits, adversarial_target)
        
        # Total loss
        loss = reconstruction_loss + 0.1 * adversarial_loss_value
        
        # print(reconstructed_logits.view(-1, reconstructed_logits.size(-1))[0], input_ids.size())
        # loss = criterion(reconstructed_ids*attention_mask, input_ids)
        # loss = criterion(logits, input_ids)
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
    
    classifier = IntentModel(args, tokenizer, target_size=60).to(device)
    classifier_pretrained = load_model(classifier, "text_classifier.pth")
    if not classifier_pretrained:
        baseline_train(args, classifier, datasets, tokenizer)    
    
    
    dataloader = get_dataloader(args, datasets['train'], split='train')  #/////////// might need to change

    # Initialize model, optimizer, and criterion
    autoencoder = Autoencoder()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    # # criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion = SequenceReconstructionLoss(len(tokenizer))
    adversarial_loss = AdversarialLoss()

    # Define device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    autoencoder.to(device)

    # Training loop
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     loss = train(autoencoder, classifier, dataloader, optimizer, criterion, device)
    #     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")