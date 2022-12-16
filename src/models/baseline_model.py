import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels, num_layers=3, layer='gcn'):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        if layer == 'gcn':
            self.conv1 = GCNConv(num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            
        elif layer == 'sageconv':
            self.conv1 = SAGEConv(num_node_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        
        self.lin = Linear(hidden_channels, num_classes)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x