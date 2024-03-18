import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


def custom_loss(labels, proj4_logits, logits, k, s1, s2):
	proj4_sorted_k_indices = np.argsort(-proj4_logits)[:k]
	proj4_topk_labels = labels[proj4_sorted_k_indices]

	sorted_k_indices = np.argsort(-ogits)[:k]
	topk_labels = labels[sorted_k_indices]

	k_intersect = len(set(proj4_topk_labels) & set(topk_labels))

	similarity = cosine_similarity(s1, s2)

	return 0.5 * (1 - k_intersect) + (1 - similarity)

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.cosine_similarity = nn.CosineSimilarity()
        self.adversarial_loss = nn.CrossEntropyLoss()

    def forward(self, generated_images, real_images, labels):
        cos_sim_loss = 1 - self.cosine_similarity(generated_images, real_images).mean()
        adv_loss = self.adversarial_loss(generated_images, labels)
        return self.alpha * cos_sim_loss + (1 - self.alpha) * adv_loss










