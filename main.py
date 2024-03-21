import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from model import Autoencoder, IntentModel
# from utils import WMT14Dataset
from datasets import load_dataset
from tqdm import tqdm as progress_bar
import torch.nn.functional as F
from utils import set_seed, setup_gpus, check_directories, WMT14Dataset, BaseInstance
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params


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
        # Zero out the log probabilities corresponding to the target class
        logs = torch.log(torch.sigmoid(pred))
        logs = logs.scatter(1, target.unsqueeze(1), 0)  # Set log probabilities of target class to 0
        # print(pred[0],logs[0], target)
        return -torch.sum(logs, dim=1) / pred.shape[0]



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
    

def train_autoenc(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    pbar = progress_bar(enumerate(dataloader), total=len(dataloader))
    for batch_idx, batch in pbar:
        # print(batch[1].size())
        input_ids = batch[0].squeeze().to(device)
        # attention_mask = input_ids != tokenizer.pad_token_id
        attention_mask = batch[1].squeeze().to(device)
        optimizer.zero_grad()
        reconstructed_logits = model(input_ids, attention_mask)
        # print(reconstructed_logits.size(), input_ids.size())
        loss = criterion(reconstructed_logits, input_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx == len(dataloader) - 1:  # Check if it's the last batch
            # Decode logits into tokenized sentences
            decoded_sequences = torch.argmax(reconstructed_logits, dim=-1)  # Shape: (batch_size, sequence_length)
            decoded_texts = [tokenizer.decode(seq.tolist(), skip_special_tokens= True) for seq in decoded_sequences]
            input_texts = [tokenizer.decode(ids.tolist(), skip_special_tokens=True) for ids in input_ids]

            # Write input and output decoded sentences to file for the last batch of the epoch
            with open("decoded_samples.txt", "a") as f:
                f.write(f"Epoch: {epoch+1}\n")
                for i in range(len(input_texts)):
                    f.write("Input Text:\n")
                    f.write(input_texts[i] + "\n\n")
                    f.write("Decoded Output Text:\n")
                    f.write(decoded_texts[i] + "\n\n")
                f.write("-------------------------------\n")
        pbar.set_postfix({"Batch Loss": loss.item(),"Total Loss": total_loss/len(dataloader)})

    return total_loss / len(dataloader)


# Define training function
def train(model, intent_classifier, dataloader, optimizer, criterion, adversarial_loss, epoch):
    model.train()
    total_loss = 0.0
    # print(len(dataloader))
    pbar = progress_bar(enumerate(dataloader), total=len(dataloader))
    
    for batch_idx, batch in pbar:        
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
        intent_logits = intent_classifier({"input_ids":decoded_sequences, "attention_mask":attention_mask})
        
        reconstruction_loss = criterion(logits, input_ids)
        
        # Adversarial loss
        # with torch.no_grad():
        adversarial_target = torch.argmax(intent_logits, dim=1)  # get the target class
        adversarial_loss_value = adversarial_loss(intent_logits, adversarial_target)
        # print(reconstruction_loss, adversarial_loss_value)

        # Total loss
        loss = reconstruction_loss + 0.01 * adversarial_loss_value.mean()
        
        # print(reconstructed_logits.view(-1, reconstructed_logits.size(-1))[0], input_ids.size())
        # loss = criterion(reconstructed_ids*attention_mask, input_ids)
        # loss = criterion(logits, input_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx >= len(dataloader) - 2:  # Check if it's the last batch
            # Decode logits into tokenized sentences
            decoded_sequences = torch.argmax(logits, dim=-1)  # Shape: (batch_size, sequence_length)
            decoded_texts = [tokenizer.decode(seq.tolist(), skip_special_tokens= True) for seq in decoded_sequences]
            input_texts = [tokenizer.decode(ids.tolist(), skip_special_tokens=True) for ids in input_ids]

            # Write input and output decoded sentences to file for the last batch of the epoch
            with open("decoded_samples.txt", "a") as f:
                f.write(f"Epoch: {epoch+1}\n")
                for i in range(len(input_texts)):
                    f.write("Input Text:\n")
                    f.write(input_texts[i] + "\n\n")
                    f.write("Decoded Output Text:\n")
                    f.write(decoded_texts[i] + "\n\n")
                f.write("-------------------------------\n")
        pbar.set_postfix({"Batch Loss": loss.item(),"Total Loss": total_loss/len(dataloader)})
    return total_loss / len(dataloader)


# Define test function
def test(args, model, datasets, intent_classifier):
    dataloader = get_dataloader(args, datasets['test'], split='test')
    model.eval()
    intent_classifier.eval()
    # print(len(dataloader))
    pbar = progress_bar(enumerate(dataloader), total=len(dataloader))
    acc = 0
    losses = 0 
    for batch_idx, batch in pbar:        
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
        intent_logits = intent_classifier({"input_ids":decoded_sequences, "attention_mask":attention_mask})

        criterian = nn.CrossEntropyLoss()
        loss = criterian(intent_logits, labels)
        losses += loss.item()

        tem = (intent_logits.argmax(1) == labels).float().sum()
        acc += tem.item()

        if batch_idx >= len(dataloader) - 2:  # Check if it's the last batch
            # Decode logits into tokenized sentences
            decoded_sequences = torch.argmax(logits, dim=-1)  # Shape: (batch_size, sequence_length)
            decoded_texts = [tokenizer.decode(seq.tolist(), skip_special_tokens= True) for seq in decoded_sequences]
            input_texts = [tokenizer.decode(ids.tolist(), skip_special_tokens=True) for ids in input_ids]

            # Write input and output decoded sentences to file for the last batch of the epoch
            with open("decoded_test_samples.txt", "a") as f:
                # f.write(f"Epoch: {epoch+1}\n")
                for i in range(len(input_texts)):
                    f.write("Input Text:\n")
                    f.write(input_texts[i] + "\n\n")
                    f.write("Decoded Output Text:\n")
                    f.write(decoded_texts[i] + "\n\n")
                f.write("-------------------------------\n")
        pbar.set_postfix({"Batch Loss": loss.item(),"Total Loss": losses/len(dataloader)})
        
    print('test acc:', acc/len(datasets['test']*100), f'| {losses} loss ', f'|dataset split test size:', len(datasets['test']))
    return acc / len(dataloader)

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
    autoencoder = Autoencoder()
    classifier_pretrained = load_model(classifier, "text_classifier.pth")
    generator_pretrained = load_model(autoencoder, "generator.pth")
    if not classifier_pretrained:
        baseline_train(args, classifier, datasets, tokenizer)    
    
    
    dataloader = get_dataloader(args, datasets['train'], split='train')  #/////////// might need to change

    # Initialize model, optimizer, and criterion
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    # # criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion = SequenceReconstructionLoss(len(tokenizer))
    adversarial_loss = AdversarialLoss()

    # Define device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    autoencoder.to(device)
    
    # with open("all_inputs_100","rb") as f:
    #     data = pickle.load(f)
    # Initialize dataset and dataloader
    wmt14_dataset = WMT14Dataset("all_inputs_100", tokenizer=tokenizer)
    # dataloader = DataLoader(wmt14_dataset, batch_size=16, shuffle=True, collate_fn=wmt14_dataset.collate_fn)
    # print(len(wmt14_dataset))

    # Training loop
    
    
    # if not generator_pretrained:
    num_epochs = 0
    for epoch in range(num_epochs):
        loss = train_autoenc(autoencoder, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
        torch.save(autoencoder.state_dict(), "generator.pth")



    # Training loop
    num_epochs = 35
    for epoch in range(num_epochs):
        loss = train(autoencoder, classifier, dataloader, optimizer, criterion, adversarial_loss, epoch)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
        if epoch%2==0:
            torch.save(autoencoder.state_dict(), "generator.pth") # save model
                
    test(args, autoencoder, datasets, classifier)
                
