import pandas as pd
import torch
import encoders

from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

DATA_PATH = ".."
DATA_SAVE_PATH = "../out/dataset"

def load_node_csv(path, index_col, x_encoders = None, y_encoders = None, **kwargs):
	"""
	Loads and encodes the node features from a .csv file into a tensor format
	for efficient parallelisation in PyTorch.

	Parameters:
		path: path to the csv file
		index_col: the column of the index to use
		x_encoders: a dictionary mapping feature column labels in the csv to an appropriate encoder
		y_encoders: a dictionaty mapping ground truth column labels to an appropriate encoder
		**kwargs: keyword arguments to pass to the csv reader
	
	Returns a tuple of the encoded node data as a tensor, and a mapping from row indexes in the csv
	to the equivalent row of the tensor.
	"""
	# Read file
	df = pd.read_csv(path, index_col = index_col, **kwargs)

	mapping = {index : i for i, index in enumerate(df.index)}

	# Encode the feature data into a tensor
	node_xs = None
	if x_encoders is not None:
		xs = [encoder(df[col]) for col, encoder in x_encoders.items()]

		# Concatenate the encoded values to produce a tensor for use in PyTorch
		node_xs = torch.cat(xs, dim = -1)

	node_ys = None
	if y_encoders is not None:
		ys = [encoder(df[col]) for col, encoder in y_encoders.items()]

		node_ys = torch.cat(ys, dim = -1)

	return node_xs, node_ys, mapping

def load_edge_csv(path, index_col, src_col, dst_col, row_mapping, encoders = None, directed = True,
		**kwargs):
	"""
	Loads and encodes the edges (with a associated edge features) of a graph stored in a .csv into
	a tensor format for efficient parallelisatio in PyTorch.

	Parameters:
		path: path to the csv file
		index_col: the column of the index to use for the edges
		src_col: the column containing the source node
		dst_col: the column containing the destination node
		row_mapping: dictionary mapping the index of a node to the associated row
					 in the node tensor.
		encoders: a dictionary mapping column labels in the csv to an appropriate encoder
		directed: whether the edges are directed or not
		**kwargs: keyword arguments to pass to the csv reader.

	Returns a tuple of the encoded edge data as a tensor, and a mapping from row indexes in the csv
	to the equivalent row of the tensor
	"""
	# Read file
	df = pd.read_csv(path, index_col = index_col, **kwargs)

	# Apply row mapping so edge indexes match up with the correct row in the node tensor
	processed_edges = df.get([src_col, dst_col]).map(lambda x: row_mapping[x]).to_numpy()

	# Reshape edges to be correctly interpreted
	edges = torch.tensor(processed_edges).t().contiguous()

	if not directed:
		rev_edges = edges.flip([0])
		edges = torch.cat((edges, rev_edges), -1)
	
	# Encode the feature data into a tensor
	edge_features = None
	if encoders is not None:
		xs = [encoder(df[col]) for col, encoder in encoders.items()]

		# Concatenate the encoded values to produce a tensor for use in PyTorch
		edge_features = torch.cat(xs, dim = -1)
		if not directed:
			edge_features = torch.cat((edge_features, edge_features), 0)
	return edges, edge_features
	
def load_csv_data(root, device = None, num_val = 0.1, num_test = 0.2):
	"""
	Loads and preprocesses the musae graph data from csv files inside the provided root folder.

	Parameters:
		root: a file path to the location where the musae folder was downloaded to
			  (see https://snap.stanford.edu/data/facebook-large-page-page-network.html).
			  Assumes file was unzipped in root/
		device: the device for torch to use for preprocessing, if necessary
		num_val: proportion of dataset nodes to use for validation
		num_test: proportion of dataset nodes to use for testing

	Returns the preprocessed dataset for feeding to PyTorch
	"""
	
	# Create encoders
	name_encoder = encoders.NameEncoder(
		"sentence-transformers/distiluse-base-multilingual-cased-v2",
		device = device
	)

	type_encoder = encoders.PageEncoder(["government", "company", "tvshow", "politician"])

	# Extract the node data from csv
	node_xs, node_ys, row_mapping = load_node_csv(
		root + "/facebook_large/facebook_large/musae_facebook_target.csv",
		"id",
		x_encoders = {"page_name" : name_encoder},
		y_encoders = {"page_type" : type_encoder}
	)

	# Extract the edge data from csv
	edges, _ = load_edge_csv(
		root + "/facebook_large/facebook_large/musae_facebook_edges.csv",
		None,
		"id_1",
		"id_2",
		row_mapping,
		directed = False
	)

	# Create dataset
	data = Data(x = node_xs, edge_index = edges, y = node_ys)

	# Split into training, validation and testing nodes
	transform = RandomNodeSplit(num_val = num_val, num_test = num_test)
	
	return transform(data)


def save_preprocessed_data(data, path):
	"""
	"""
	torch.save(data, path)
