import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_plot(prediction, true_colour):
	"""
	"""
	z = TSNE(n_components = 2, verbose = 1).fit_transform(prediction.detach().cpu().numpy())
	
	plt.scatter(z[:,0], z[:, 1], c = true_colour, cmap = "Accent", alpha = 0.3)
	plt.show()
	return None
