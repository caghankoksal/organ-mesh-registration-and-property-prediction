import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, Linear
from torch_geometric.nn import global_max_pool, global_mean_pool
from itertools import tee
from src.models.point_transformer_block import PointTransformerLayer

def pairwise(iterable):
    """Iterate over all pairs of consecutive items in a list.
    Notes
    -----
        [s0, s1, s2, s3, ...] -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def get_mlp_layers(channels: list, activation, output_activation=nn.Identity):
    """Define basic multilayered perceptron network."""
    layers = []
    *intermediate_layer_definitions, final_layer_definition = pairwise(channels)

    for in_ch, out_ch in intermediate_layer_definitions:
        intermediate_layer = nn.Linear(in_ch, out_ch)
        layers += [intermediate_layer, activation()]

    layers += [nn.Linear(*final_layer_definition), output_activation()]
    #print('Output activation ',output_activation)
    return nn.Sequential(*layers)

def get_pt_layer():
    layers = nn.ModuleList()

    attn = PointTransformerLayer(
    dim = 128,
    pos_mlp_hidden_dim = 64,
    attn_mlp_hidden_mult = 4,
    num_neighbors = 3
    )

    feats = torch.randn(1, 2048, 128)
    pos = torch.randn(1, 2048, 3)
    mask = torch.ones(1, 2048).bool()

    attn(feats, pos, mask = mask)
    layers.append(attn)

    return nn.ModuleList(layers)


class PT(torch.nn.Module):
    def __init__(self, in_features, num_classes, hidden_channels, num_layers=3, layer='gcn',
                 use_input_encoder=True, encoder_features=128, apply_batch_norm=True,
                 apply_dropout_every=True, task='sex_prediction', use_scaled_age=False, dropout = 0):

        super().__init__()

        assert task in ['age_prediction']
        torch.manual_seed(12345)

        self.fc = torch.nn.ModuleList()
        self.pt = torch.nn.ModuleList()
        self.task = task
        self.layer_type = layer
        self.use_input_encoder = use_input_encoder
        self.apply_batch_norm = apply_batch_norm
        self.dropout = dropout
        self.apply_dropout_every = apply_dropout_every
        self.use_scaled_age = use_scaled_age
        if self.use_input_encoder :
            self.input_encoder = get_mlp_layers(
                channels=[in_features, encoder_features],
                activation=nn.ELU,
            )
            in_features = encoder_features

        for i in range(num_layers):
            self.pt.append(get_pt_layer())

        self.fc.append(Linear(encoder_features, 1))

        print(self.input_encoder)
        print(self.pt)
        print(self.fc)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input_encoder(x)

        for i in range(len(self.pt)):
            if i == 0:
                x = self.pt[i](x)
            else:
                x = self.pt[i](x)
                x = self.input_encoder(x)
                x = global_max_pool(x, batch)

        # 2. Readout layer
          # [batch_size, hidden_channels]
        x = global_mean_pool(x, batch)
        
        x = self.input_encoder(x)

        x = self.fc(x)
        if self.use_scaled_age or self.task =='sex_prediction':
            x = torch.nn.Sigmoid()(x)
        
        return x