import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Autoencoder(nn.Module):
    def __init__(self, pretrained_model_name):
        super(Autoencoder, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=8),
            num_layers=6
        )
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vocab_size = self.tokenizer.vocab_size

    def forward(self, input_ids, attention_mask):
        # Encode input sequence
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoded_sequence = encoder_output.last_hidden_state
        
        # Decode encoded sequence autoregressively
        decoded_sequence = self.decode(encoded_sequence)
        
        return decoded_sequence

    def decode(self, encoded_sequence):
        batch_size, seq_length, hidden_size = encoded_sequence.size()
        decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long).to(encoded_sequence.device)
        
        # Autoregressive decoding
        for step in range(seq_length):
            decoder_output = self.decoder(
                memory=encoded_sequence,
                tgt=self.tokenizer(decoder_input_ids, return_tensors='pt').input_ids,
            )
            next_token_logits = decoder_output[0][:, -1, :]  # logits for the next token
            next_token_id = next_token_logits.argmax(-1)     # get the index of the highest probability token
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(-1)], dim=-1)
        
        return decoder_input_ids[:, 1:]  # remove the initial padding token

#################### DEFINE MODELS AND CUSTOM LOSS ####################
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 784)  # Output size is the same as input size (MNIST images)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    
class DigitClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)  # Output size is 10 for 10 classes (digits)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.activation(x)
        return x

class CifarClassifier(nn.Module):
    def __init__(self):
        super(CifarClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    