import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.nn import functional as F

class IntentEncoder(nn.Module):
    def __init__(self, args, tokenizer, target_size=60, feat_dim=768):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.encoder.resize_token_embeddings(len(self.tokenizer)) 
        self.target_size = target_size
        self.dropout = nn.Dropout(args.drop_rate)
        self.head = nn.Linear(feat_dim, args.embed_dim)

    def forward(self, inputs):
        out1 = self.encoder(**inputs).last_hidden_state[:, 0, :]
        out2 = self.dropout(out1)
        return F.normalize(self.head(out2), p=2, dim=-1)
    
class IntentClassifier(nn.Module):
    def __init__(self, args, intent_encoder, target_size=60):
        super().__init__()
        input_dim = args.embed_dim
        self.top = nn.Linear(input_dim, args.hidden_dim)
        self.relu = nn.ReLU()
        self.bottom = nn.Linear(args.hidden_dim, target_size)
        self.encoder = intent_encoder

    def forward(self, inputs):
        hidden = self.encoder(inputs)
        middle = self.relu(self.top(hidden))
        logit = self.bottom(middle)
        return logit
    

class Generator(nn.Module):
    def __init__(self, args, tokenizer, target_size=60, feat_dim=768):
        super(Autoencoder, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = BertModel.from_pretrained("bert-base_uncased")
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.target_size = target_size
        self.head = nn.Linear(feat_dim, args.embed_dim)

    def forward(self, inputs):
        out1 = self.encoder(**inputs).last_hidden_state[:, 0, :]
        return self.head(out1)



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.decoder_lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=1, batch_first=True)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, 768)
        self.fc = nn.Linear(768, self.vocab_size)

    def forward(self, input_ids, attention_mask):
        # Encode input sequence
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoded_sequence = encoder_output.last_hidden_state
        
        # Decode encoded sequence using LSTM
        decoded_sequence = self.decode(encoded_sequence)
        
        return decoded_sequence

    def decode(self, encoded_sequence):
        batch_size, seq_length, hidden_size = encoded_sequence.size()
        decoder_input_ids = torch.full((batch_size, 1), self.tokenizer.cls_token_id, dtype=torch.long, device=encoded_sequence.device)

        # Initialize LSTM hidden state and cell state
        h_0 = torch.zeros(1, batch_size, hidden_size, device=encoded_sequence.device)
        c_0 = torch.zeros(1, batch_size, hidden_size, device=encoded_sequence.device)

        # Autoregressive decoding using LSTM
        logits = []
        for step in range(seq_length):
            decoder_input_emb = self.embedding(decoder_input_ids)
            decoder_output, (h_t, c_t) = self.decoder_lstm(decoder_input_emb, (h_0, c_0))
            decoder_output = self.fc(decoder_output)
            logits.append(decoder_output)
            next_token_logits = decoder_output[:, -1, :]  # logits for the next token
            next_token_id = next_token_logits.argmax(-1) # get the index of the highest probability token
            decoder_input_ids = next_token_id.unsqueeze(-1)  # Update input_ids for the next time step
            # Update hidden state and cell state for next time step
            h_0 = h_t
            c_0 = c_t
        
        logits = torch.cat(logits, dim=1)  # Concatenate logits along the sequence length dimension
        return logits
