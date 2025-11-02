import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import modules
from dataset import load_csv_data, DATA_PATH
from train import NUM_HIDDEN_CHANNELS, SAVE_PATH

PLOT_SAVE_PATH = "../assets/tsne_plot.png"

def tsne_plot(prediction, true_colour):
	"""
	Produces a TNSE embedding of the predictions of the model, coloured by the
	ground truth values of the points.

	Parameters:
		prediction: the predicted classifications of the model
		true_colour: the actual classification of the datapoints in a numeric form
	"""
	z = TSNE(n_components = 2, verbose = 1, perplexity = 30).fit_transform(prediction.detach().cpu().numpy())
	
	plt.scatter(z[:,0], z[:, 1], c = true_colour, cmap = "Accent")
	plt.title("TSNE Plot of Model Predictions with Ground Truth Values Shown")
	plt.colorbar()
	return plt

def load_gcn_model(num_features, num_categories, path):
	"""
	Loads a GCN state dict and produces a model ready for evaluation.

	Parameters:
		num_features: the number of features used in the trained model
		num_categories: the number of categories used in the trained model
		path: the path to the trained model
	"""
	model = modules.GCN(num_features, num_categories, NUM_HIDDEN_CHANNELS)
	model.load_state_dict(torch.load(SAVE_PATH, weights_only = True))
	model.eval()
	return model

if __name__ == "__main__":
	# Load data
	data = load_csv_data(DATA_PATH)

	# Load the model
	model = load_gcn_model(data.num_features,  data.y.size()[1], SAVE_PATH)

	
	plot = tsne_plot(model(data.x, data.edge_index), true_colour = data.y.argmax(dim = -1))
	plot.savefig(PLOT_SAVE_PATH)
	plot.show()
