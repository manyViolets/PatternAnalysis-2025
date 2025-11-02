import torch
import torch.nn.functional as tnf
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
	"""
    A graph convolutional network layer.
	"""
	def __init__(self, num_features, num_categories, hidden_channels):
		"""
		Parameters:
			num_features: the number of node and edge features of the dataset
			num_categories: the number of ground truth categories
			hidden_channels: the number of hidden channels to use for the
							 inner layer
		"""
		super().__init__()
		self.conv1 = GCNConv(num_features, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, num_categories)
		

	def forward(self, x, edge_index):
		"""
		Training step of the model.

		Parameters:
			x: node features
			edge_index: edge data
		"""
		x = self.conv1(x, edge_index)
		x = x.relu()
		x = tnf.dropout(x, p = 0.5, training = self.training)
		x = self.conv2(x, edge_index)
		return x
