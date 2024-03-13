from transformers import MarianTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch


class BaseInstance(object):
    def __init__(self, embed_data, text):
        self.embedding = embed_data['input_ids']
        self.input_mask = embed_data['attention_mask']
        self.text = text


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

    def __len__(self):
        return len(self.data['en'])  # Assuming equal number of sentences for both languages

    def __getitem__(self, idx):
        en_instance = self.data['en'][idx]
        de_instance = self.data['de'][idx]

        input_ids_en = en_instance.embedding
        input_ids_de = de_instance.embedding
        attention_mask_en = en_instance.input_mask
        attention_mask_de = de_instance.input_mask

        return input_ids_en, input_ids_de, attention_mask_en, attention_mask_de
    
    def collate_func(self, batch):
        # print(len(list(zip(*batch))))
        input_ids_en, input_ids_de, attention_mask_en, attention_mask_de = zip(*batch)
    
        # Pad sequences within batch to the same length
        input_ids_en_padded = pad_sequence([t.squeeze() for t in input_ids_en], batch_first=True, padding_value=0)
        input_ids_de_padded = pad_sequence([t.squeeze() for t in input_ids_de], batch_first=True, padding_value=0)
        attention_mask_en_padded = pad_sequence([t.squeeze() for t in attention_mask_en], batch_first=True, padding_value=0)
        attention_mask_de_padded = pad_sequence([t.squeeze() for t in attention_mask_de], batch_first=True, padding_value=0)
        
        return input_ids_en_padded, input_ids_de_padded, attention_mask_en_padded, attention_mask_de_padded

