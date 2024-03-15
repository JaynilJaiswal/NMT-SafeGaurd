from sklearn.metrics.pairwise import cosine_similarity

def custom_loss(labels, proj4_logits, logits, k, s1, s2):
	proj4_sorted_k_indices = np.argsort(-proj4_logits)[:k]
	proj4_topk_labels = labels[proj4_sorted_k_indices]

	sorted_k_indices = np.argsort(-ogits)[:k]
	topk_labels = labels[sorted_k_indices]

	k_intersect = len(set(proj4_topk_labels) & set(topk_labels))

	similarity = cosine_similarity(s1, s2)

	return 0.5 * (1 - k_intersect) + (1 - similarity)











