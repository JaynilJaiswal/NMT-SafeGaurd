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