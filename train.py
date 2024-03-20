import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm as progress_bar

from dataloader import get_dataloader, prepare_inputs, prepare_features


class Loss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(Loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        logs = torch.log(pred)
        for i in range(target.shape[0]):
            logs[i, target[i].item()] = 0
        return -torch.sum(logs, dim=1) / (pred.shape[1])

def run_eval(args, model, datasets, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)
    acc = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch)
        logits = model(inputs)
        acc += (logits.argmax(1) == labels).float().sum().item()
    print(f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split]))

def train(args, model, datasets):
    criterion = Loss(temperature=args.temperature)
    train_dataloader = get_dataloader(args, datasets['train'], split='train')
    class_optimizer = AdamW(model.encoder.parameters(), lr=1e-5)
    class_scheduler = get_linear_schedule_with_warmup(class_optimizer, num_warmup_steps=int(0.1*args.n_epochs), num_training_steps=args.n_epochs)

    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch)
            num_views = 2 
            embeddings_list = []
            for _ in range(num_views):
                embeddings = model.encoder(inputs)
                embeddings_list.append(embeddings)
            augmented_embeddings = torch.stack(embeddings_list, dim=1)
            loss = criterion(augmented_embeddings, labels)

            # Backward and optimize
            loss.backward()
            class_optimizer.step()
            model.encoder.zero_grad()
            losses += loss.item()

        class_scheduler.step()  # Update learning rate schedule
        print('epoch', epoch_count, '| losses:', losses)
    
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    class_criterion = nn.CrossEntropyLoss()
    class_optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    class_scheduler = get_linear_schedule_with_warmup(class_optimizer, num_warmup_steps=int(0.1*args.n_epochs), num_training_steps=args.n_epochs)

    # Training loop for classification
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch)
            logits = model(inputs)
            loss = class_criterion(logits, labels)

            # Backward and optimize
            loss.backward()
            class_optimizer.step()
            model.zero_grad()
            losses += loss.item()
        class_scheduler.step()  # Update learning rate schedule
        run_eval(args=args, model=model, datasets=datasets)
        print('Classification epoch', epoch_count, '| losses:', losses)

def train_generator(args, generator, classifier, encoder, tokenizer, datasets):
    criterion = AdversarialLoss()
    train_dataloader = get_dataloader(args, datasets['train'], split='train')
    gen_optimizer = AdamW(generator.encoder.parameters(), lr=1e-5)
    gen_scheduler = get_linear_schedule_with_warmup(gen_optimizer, num_warmup_steps=int(0.1*args.n_epochs), num_training_steps=args.n_epochs)

    generator.train()
    classifier.eval()
    encoder.eval()

    for epoch_count in range(args.n_epochs):
        losses = 0
        pb = progress_bar(enumerate(train_dataloader), total=len(train_dataloader))
        for step, batch in pb:
            generator.zero_grad()

            inputs, labels = prepare_inputs(batch)
            generated_inputs, masks = generator(inputs)

            generated_inputs = torch.argmax(generated_inputs, dim=-1)
            for i in range(len(generated_inputs)): # apply mask to generated input
                generated_inputs[i, masks[i] == 0] = 0
            
            generated_inputs = {
                'input_ids': generated_inputs,
                'token_type_ids': torch.zeros_like(generated_inputs),
                'attention_mask': masks
            }
            
            # print(tokenizer.batch_decode(generated_inputs['input_ids'], skip_special_tokens=True)) # print out sentences

            orig_embed = encoder(inputs, dropout=False)
            generated_embed = encoder(generated_inputs, dropout=False)

            cosine_similarity = F.cosine_similarity(orig_embed, generated_embed, dim=1).mean()  # Cosine similarity loss
            adversarial_loss = criterion(classifier(generated_inputs), labels)                  # Adversarial loss from classifier

            # Total generator loss
            generator_loss = (1-cosine_similarity) + 0.01*torch.mean(adversarial_loss)
            generator_loss.backward()
            gen_optimizer.step()

            losses += generator_loss.item()
            pb.set_postfix({'Generator Loss': losses / len(train_dataloader)})
        gen_scheduler.step()  # Update learning rate schedule