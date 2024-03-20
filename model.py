import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


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
        decoded_sequence = self.decode(encoded_sequence, input_ids)
        
        return decoded_sequence

    def decode(self, encoded_sequence, target_sequence):
        batch_size, seq_length, hidden_size = encoded_sequence.size()
        decoder_input_ids = torch.full((batch_size, 1), self.tokenizer.cls_token_id, dtype=torch.long, device=encoded_sequence.device)

        # Initialize LSTM hidden state and cell state
        h_0 = torch.zeros(1, batch_size, hidden_size, device=encoded_sequence.device)
        c_0 = torch.zeros(1, batch_size, hidden_size, device=encoded_sequence.device)

        # Autoregressive decoding using LSTM
        logits = []
        decoder_input_emb = self.embedding(decoder_input_ids)
        for step in range(seq_length):
            decoder_output, (h_t, c_t) = self.decoder_lstm(decoder_input_emb, (h_0, c_0))
            decoder_output = self.fc(decoder_output)
            logits.append(decoder_output)
            # next_token_logits = decoder_output[:, -1, :]  # logits for the next token
            # next_token_id = next_token_logits.argmax(-1) # get the index of the highest probability token
            # decoder_input_ids = next_token_id.unsqueeze(-1)  # Update input_ids for the next time step
            # Use teacher forcing: Use the next token from the target sequence as input to the decoder
            decoder_input_emb = self.embedding(target_sequence[:, step].unsqueeze(1))  # Get next token embedding      
            # Update hidden state and cell state for next time step
            h_0 = h_t
            c_0 = c_t
        
        logits = torch.cat(logits, dim=1)  # Concatenate logits along the sequence length dimension
        return logits
    
class IntentModel(nn.Module):
    def __init__(self, args, tokenizer, target_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_setup(args)
        self.target_size = target_size

        # task1: add necessary class variables as you wish.
        self.args = args
        print(self.args)

        # task2: initilize the dropout and classify layers
        self.dropout = nn.Dropout(self.args.drop_rate)
        self.classify = Classifier(args,target_size=target_size)
    
    def model_setup(self, args):
        print(f"Setting up {args.model} model")

        # task1: get a pretrained model of 'bert-base-uncased'
        #self.encoder = BertModel.from_pretrained(...)
        self.encoder = BertModel.from_pretrained('bert-base-uncased')

        self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check
        #self.encoder.resize_token_embeddings(len(self.tokenizer)) 

    def forward(self, inputs):
        """
        task1: 
            feeding the input to the encoder, 
        task2: 
            take the last_hidden_state's <CLS> token as output of the
            encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
        task3:
            feed the output of the dropout layer to the Classifier which is provided for you.
        """

        out1 = self.encoder(**inputs).last_hidden_state[ :, 0 , :]  # (batch_size, seq len, hidden size)
        # print(out1.size())

        return self.classify(self.dropout(out1))


class Classifier(nn.Module):
    def __init__(self, args, target_size):
        super().__init__()
        input_dim = args.embed_dim
        self.top = nn.Linear(input_dim, args.hidden_dim)
        self.relu = nn.ReLU()
        self.bottom = nn.Linear(args.hidden_dim, target_size)

    def forward(self, hidden):
        middle = self.relu(self.top(hidden))
        logit = self.bottom(middle)
        return logit
