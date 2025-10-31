import pandas as pd
import torch

def load_node_csv(path, index_col, encoders = None, **kwargs):
	"""
	Loads and encodes the node features from a .csv file into a tensor format
	for efficient parallelisation in PyTorch.

	Parameters:
		path: path to the csv file
		index_col: the column of the index to use
		encoders: a dictionary mapping column labels in the csv to an appropriate encoder
		**kwargs: keyword arguments to pass to the csv reader
	
	Returns a tuple of the encoded node data as a tensor, and a mapping from row indexes in the csv
	to the equivalent row of the tensor.
	"""
	# Read file
	df = pd.read_csv(path, index_col = index_col, **kwargs)

	mapping = {index : i for i, index in enumerate(df.index)}

	# Encode the data into a tensor
	nodes = None
	if encoders is not None:
		xs = [encoder(df[col]) for col, encoder in encoders.items()]

		# Concatenate the encoded values to produce a tensor for use in PyTorch
		nodes = torch.cat(xs, dim = -1)
	
	return nodes, mapping

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

	edges = torch.tensor(processed_edges)

	if not directed:
		rev_edges = torch.tensor(processed_edges.copy()[:-1])
		edges = torch.cat((edges, rev_edges), 0)
	
	# Encode the feature data into a tensor
	edge_features = None
	if encoders is not None:
		xs = [encoder(df[col]) for col, encoder in encoders.items()]

		# Concatenate the encoded values to produce a tensor for use in PyTorch
		edge_features = torch.cat(xs, dim = -1)
		if not directed:
			edge_features = torch.cat((edge_features, edge_features), 0)
	return edges, edge_features
	
