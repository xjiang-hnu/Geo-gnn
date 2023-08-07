import torch
from torch.nn import  Embedding, Linear,Parameter

from torch_scatter import scatter

from torch_geometric.nn.inits import glorot_orthogonal

class triplet_to_edge_layer(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size,
                 num_spherical, 
                 act):
        super().__init__()
        self.act = act
        self.lin_sbf0 = Linear(num_spherical, num_spherical, bias=False)
        self.lin_sbf = Linear(num_spherical, int_emb_size, bias=False)

        self.lin_e1 = Linear(2*hidden_channels,hidden_channels)

        # Embedding projections for interaction triplets:
        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):

        glorot_orthogonal(self.lin_sbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_e1.weight, scale=2.0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)


    def forward(self, edge_attr, sbf,idx_kj, idx_ji):
        # Initial transformation:

        # Down project embedding and generating triple-interactions:
        x = self.act(self.lin_down(edge_attr))

        # Transform via 2D spherical basis:
        sbf = self.lin_sbf0(sbf)
        sbf = self.lin_sbf(sbf)
        # sbf = self.lin_sbf2(sbf)
        x = x[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings:
        x = scatter(x, idx_ji, dim=0, dim_size=edge_attr.size(0))
        x = self.act(self.lin_up(x))
        x = self.act(self.lin_e1(torch.cat([edge_attr,x], dim =-1)))


        return x + edge_attr