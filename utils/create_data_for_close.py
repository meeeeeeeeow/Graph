import numpy as np
import torch

from torch_geometric.data import Data
from .preprocessing_for_close import build_graph

def get_graph_from_svg(file_path):
    G_close, edge_type = build_graph(file_path)
    data = create_pyg_data(G_close, edge_type)
    
    return data

def create_pyg_data(G_close, edge_type):
    # connected relationship list; edges.shape=(m*2, 2)
    m = len(G_close.edges)
    edges = np.zeros([2*m, 2]).astype(np.int64)
    edge_attr = np.zeros([2*m, 1]).astype(np.int64)  # add edge attributes
    edge_type_id = {"self": 0,
                    "group": 1,
                    "contain": 2,
                    "overlap": 3,}

    for e,(s,t) in enumerate(G_close.edges):
        edges[e, 0] = s
        edges[e, 1] = t

        edges[m+e, 0] = t
        edges[m+e, 1] = s
        
        edge_attr[e] = edge_type_id[edge_type[(s, t)]]
        edge_attr[m+e] = edge_type_id[edge_type[(s, t)]]
        
    edges = torch.Tensor(np.transpose(edges)).type(torch.long)
    edge_attr = torch.Tensor(edge_attr).type(torch.long)
    
    # nodes' features; x.shape=(num_node, num_features)
    n = len(G_close.nodes)
    x = np.zeros([n, 5]).astype(np.float32)
    y = np.zeros([n, 3]).astype(np.float32)
    for i in G_close.nodes:
        y[i, :] = G_close.nodes[i]["features"]["rgb"].tolist()  # RGB, ground truth, shape=(num_node, 3)
        G_close.nodes[i]["features"]["rgb"] = np.array([0, 0, 0])  # init RGB=(0, 0, 0)
        
        f = [0, 0, 0]
        f += G_close.nodes[i]["features"]["pos"].tolist()
        f += G_close.nodes[i]["features"]["v"].tolist()
        f += G_close.nodes[i]["features"]["e"].tolist()
        x[i, :] = f
        
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    
    # represent one graph
    data = Data(x=x, edge_index=edges, edge_attr=edge_attr, y=y)
    
    return data