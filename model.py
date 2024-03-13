import torch
from transformers import MarianMTModel, DistilBertForSequenceClassification, DistilBertTokenizer,  AdamW, get_linear_schedule_with_warmup


class CustomMTModel(torch.nn.Module):
    def __init__(self, device):
        super(CustomMTModel, self).__init__()
        self.marian_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        self.marian_model.to(device)

    def forward(self, en_input, en_mask, de_input, de_mask):
        outputs = self.marian_model(input_ids=en_input, attention_mask=en_mask, decoder_input_ids = de_input, decoder_attention_mask = de_mask)
        return outputs

class CustomDiscriminator(torch.nn.Module):
    def __init__(self, device, num_labels=2):
        super(CustomDiscriminator, self).__init__()
        # Initialize a DistilBERT model for sequence classification
        self.distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=num_labels)
        self.distilbert_model.to(device)
        # Initialize a tokenizer for DistilBERT
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        self.device = device

    def forward(self, text_sequences):
        # Tokenize the input sequences and convert them to tensors
        inputs = self.tokenizer(text_sequences, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = inputs.to(self.device)

        # Pass the inputs to the DistilBERT model
        outputs = self.distilbert_model(**inputs)

        return outputs.logits


class TranslationGAN(torch.nn.Module):
    def __init__(self, device, generator, discriminator):
        super(TranslationGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.device = device

    def forward(self, src_input, src_mask, tgt_input, tgt_mask):
        # Generate fake translations
        gen_outputs = self.generator(src_input, src_mask, tgt_input, tgt_mask)
        fake_translations = gen_outputs.logits.argmax(-1)

        # Discriminate real and fake translations
        real_discrimination = self.discriminator(tgt_input)
        fake_discrimination = self.discriminator(fake_translations)

        return real_discrimination, fake_discrimination







