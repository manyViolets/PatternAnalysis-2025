import torch
from torch_geometric.data import Data
import pandas as pd
import src.csv_loading as csv_loading
import src.encoders as encoders

# File location of the data
root = "."

# Create encoders
name_encoder = encoders.NameEncoder("sentence-transformers/distiluse-base-multilingual-cased-v2")

type_encoder = encoders.PageEncoder(["government", "company", "tvshow", "politician"])

# extract the node data from csv
nodes, row_mapping = csv_loading.load_node_csv(
	root + "/facebook_large/facebook_large/musae_facebook_target.csv",
	"id",
	encoders = {
	"page_name" : name_encoder,
	"page_type" : type_encoder}
)

# extract the edge data from csv
edges, _ = csv_loading.load_edge_csv(
	root + "/facebook_large/facebook_large/musae_facebook_edges.csv",
	None,
	"id_1",
	"id_2",
	row_mapping
)

data = Data(x = nodes, edge_index = edges)

print(data)
