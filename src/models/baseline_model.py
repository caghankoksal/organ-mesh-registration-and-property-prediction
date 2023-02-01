import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, LayerNorm, Linear
from torch_geometric.nn import global_mean_pool
from src.models.fsgn_model import get_mlp_layers



def get_gnn_layers(n_layers: int, hidden_channels: int, num_inp_features:int, 
                 gnn_layer, activation=nn.ReLU, normalization=None, dropout = None):
    """Creates GNN layers"""
    layers = nn.ModuleList()
   
    for i in range(n_layers):
        # First GNN layer
        if i == 0:
            layer = gnn_layer(num_inp_features, hidden_channels)
        else:
            layer = gnn_layer(hidden_channels, hidden_channels)
        
        if normalization is not None:
            norm_layer = normalization(hidden_channels)

        layers += [layer, activation(), norm_layer ]

    return nn.ModuleList(layers)



class GNN(torch.nn.Module):
    def __init__(self, in_features, num_classes, hidden_channels, num_layers=3, layer='gcn',
                 use_input_encoder=True, encoder_features=128, apply_batch_norm=True,
                 apply_dropout_every=True, task='sex_prediction', use_scaled_age=False):
        super(GNN, self).__init__()

        assert task in ['age_prediction', 'sex_prediction']
        torch.manual_seed(12345)
        
        self.fc = torch.nn.ModuleList()
        self.task = task
        self.layer_type = layer
        self.use_input_encoder = use_input_encoder
        self.apply_batch_norm = apply_batch_norm
        self.apply_dropout_every = apply_dropout_every
        self.use_scaled_age = use_scaled_age
        if self.use_input_encoder :
            self.input_encoder = get_mlp_layers(
                channels=[in_features, encoder_features],
                activation=nn.ELU,
            )
            in_features = encoder_features

        if layer == 'gcn':
            self.layers = get_gnn_layers(num_layers, hidden_channels, num_inp_features=in_features,
                                        gnn_layer=GCNConv,activation=nn.ReLU,normalization=LayerNorm )
        elif layer == 'sageconv':
            self.layers = get_gnn_layers(num_layers, hidden_channels,in_features,
                                        gnn_layer=SAGEConv,activation=nn.ReLU,normalization=LayerNorm )
        elif layer == 'gat':
            self.layers = get_gnn_layers(num_layers, hidden_channels,in_features,
                                        gnn_layer=GATConv,activation=nn.ReLU,normalization=LayerNorm )
            self.fc.append(Linear(hidden_channels, 128))
            self.fc.append(Linear(128, 128))
            self.fc.append(Linear(128, 64))

        #if apply_batch_norm:
        #    self.batch_layers = nn.ModuleList(
        #        [nn.BatchNorm1d(hidden_channels) for i in range(num_layers)]
        #    )


        if task == 'sex_prediction':
            self.pred_layer = Linear(hidden_channels, num_classes)
        elif task == 'age_prediction':
            if layer == 'gat':
                self.pred_layer = Linear(64, 1)
            else:    
                self.pred_layer = Linear(hidden_channels, 1)



    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.use_input_encoder:
            x = self.input_encoder(x)

        for i, layer in enumerate(self.layers):
            # Each GCN consists 3 modules GCN -> Activation ->  Normalization 
            # GCN send edge index
            if i% 3 == 0:
                x = layer(x, edge_index)
            else:
                x = layer(x)

            if self.apply_dropout_every:
                x = F.dropout(x, p=0.5, training=self.training)
                

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)

        if self.layer_type == 'gat':
            for i in range(len(self.fc)):
               x = self.fc[i](x)
               x = torch.tanh(x)
               x = F.dropout(x, p=0.3, training=self.training)
            x = self.pred_layer(x)
        else:
            x = self.pred_layer(x)
        if self.use_scaled_age or self.task =='sex_prediction':
            x = torch.nn.Sigmoid()(x)
        
        return x