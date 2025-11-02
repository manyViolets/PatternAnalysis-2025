import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_plot(prediction, true_colour):
	"""
	Produces a TNSE embedding of the predictions of the model, coloured by the
	ground truth values of the points.

	Parameters:
		prediction: the predicted classifications of the model
		true_colour: the actual classification of the datapoints in a numeric form
	"""
	z = TSNE(n_components = 2, verbose = 1, perplexity = 30).fit_transform(prediction.detach().cpu().numpy())
	
	plt.scatter(z[:,0], z[:, 1], c = true_colour, cmap = "Accent", alpha = 0.1)
	plt.show()
	return None
