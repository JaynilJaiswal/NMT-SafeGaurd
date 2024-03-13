import torch
from transformers import MarianMTModel


class CustomMTModel(torch.nn.Module):
    def __init__(self, device):
        super(CustomMTModel, self).__init__()
        self.marian_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        self.marian_model.to(device)

    def forward(self, en_input, en_mask, de_input, de_mask):
        outputs = self.marian_model(input_ids=en_input, attention_mask=en_mask, decoder_input_ids = de_input, decoder_attention_mask = de_mask)
        return outputs
