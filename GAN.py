from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from model import *
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle


# Initialize dataset and dataloader
custom_dataset = CustomDataset(data)
dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True, collate_fn=custom_dataset.collate_func)

# Initialize your models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = CustomMTModel(device)
discriminator = CustomDiscriminator(device)

# Instantiate the GAN
translation_gan = TranslationGAN(device, generator, discriminator)

# Define optimizers and learning rates
gen_optimizer = AdamW(generator.parameters(), lr=1e-5)
disc_optimizer = AdamW(discriminator.parameters(), lr=1e-5)

# Define loss function
criterion = torch.nn.BCEWithLogitsLoss()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    for batch_idx, batch in enumerate(dataloader):
        inputs_en, inputs_de, mask_en, mask_de = [b.to(device) for b in batch]
        
        # Generate fake translations
        with torch.no_grad():
            fake_outputs = generator(inputs_en, mask_en, inputs_de[:, :-1], mask_de[:, :-1])
            fake_translations = fake_outputs.logits.argmax(-1)
        
        # Train discriminator on real translations
        disc_optimizer.zero_grad()
        real_preds = discriminator(inputs_de[:, 1:].contiguous())
        real_loss = criterion(real_preds, torch.ones_like(real_preds))
        real_loss.backward()

        # Train discriminator on fake translations
        fake_preds = discriminator(fake_translations)
        fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))
        fake_loss.backward()
        disc_optimizer.step()

        # Train generator
        gen_optimizer.zero_grad()
        gen_outputs = generator(inputs_en, mask_en, inputs_de[:, :-1], mask_de[:, :-1])
        gen_translations = gen_outputs.logits.argmax(-1)
        gen_preds = discriminator(gen_translations)
        gen_loss = criterion(gen_preds, torch.ones_like(gen_preds))
        gen_loss.backward()
        gen_optimizer.step()

        # Print progress
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], D Loss: {(real_loss.item() + fake_loss.item()) / 2:.4f}, G Loss: {gen_loss.item():.4f}")

# Save trained models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")







