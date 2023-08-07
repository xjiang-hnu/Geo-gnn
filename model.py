import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.resolver import activation_resolver
from torch_sparse import SparseTensor

from utils import EmbeddingBlock, GaussianSmearing, SphericalBasisLayer
from encoder_layer import triplet_to_edge_layer
from aggeration_layer import edge_to_node_layer,none_to_node_layer
from mlp import MLP


class Geo_GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, act = 'swish',num_gaussians: int = 128, num_sph = 16, cutoff: float = 5.0, learn_msg_scale=True, aggr='mean'):
        super().__init__()

        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        
        act = activation_resolver(act)
        self.embedding = EmbeddingBlock(num_gaussians,hidden_channels,act)
        self.rbf = GaussianSmearing(0.0, cutoff, num_gaussians)
        #####
        self.sbf = SphericalBasisLayer(num_sph)

        self.interaction_blocks = triplet_to_edge_layer(hidden_channels, hidden_channels//2, num_sph,
                            act=act)
        self.interaction_blocks2 = triplet_to_edge_layer(hidden_channels, hidden_channels//2, num_sph,
                            act=act)
        self.interaction_blocks3 = triplet_to_edge_layer(hidden_channels, hidden_channels//2, num_sph,
                              act=act)

        # First aggeration layer in the EmbeddingBlock 
        self.agg_2 = edge_to_node_layer(hidden_channels, dim = hidden_channels)
        self.agg_3 = none_to_node_layer(MLP([2* hidden_channels, hidden_channels,hidden_channels]), aggr)

        self.mlp = MLP([hidden_channels,hidden_channels//2, out_channels], dropout=0.0)

        self.reset_parameters()
    def reset_parameters(self):
      self.embedding.reset_parameters()
      self.mlp.reset_parameters()
      self.agg_2.reset_parameters()
      self.agg_3.reset_parameters()
      self.interaction_blocks.reset_parameters()
      self.interaction_blocks2.reset_parameters()
      self.interaction_blocks3.reset_parameters()

    def triplets(self, edge_index, num_nodes):
      row, col = edge_index  # j->i

      value = torch.arange(row.size(0), device=row.device)
      adj_t = SparseTensor(row=col, col=row, value=value,
                            sparse_sizes=(num_nodes, num_nodes))
      adj_t_row = adj_t[row]
      num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

      # Node indices (k->j->i) for triplets.
      idx_i = col.repeat_interleave(num_triplets)
      idx_j = row.repeat_interleave(num_triplets)
      idx_k = adj_t_row.storage.col()
      mask = idx_i != idx_k  # Remove i == k triplets.
      idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

      # Edge indices (k-j, j->i) for triplets.
      idx_kj = adj_t_row.storage.value()[mask]
      idx_ji = adj_t_row.storage.row()[mask]

      # edge_index_angle = torch.vstack([idx_kj,idx_ji])
      return col, row, idx_i, idx_j, idx_k, idx_kj,idx_ji
    def forward(self, data):
        z, edge_index,edge_pos = data.x.to(torch.long), data.edge_index.to(torch.long), data.edge_attr.to(torch.float32)

        edge_weight = edge_pos.norm(dim = 1)

        _, _, _, _, _, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=z.size(0))

        pos_ji, pos_ki = edge_pos[idx_ji], edge_pos[idx_kj]

        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(edge_weight)
        sbf = self.sbf(angle)

        x, edge_attr = self.embedding(z,rbf,edge_index)

        edge_attr = self.interaction_blocks(edge_attr, sbf, idx_kj,idx_ji)
         #
        edge_attr = self.interaction_blocks2(edge_attr, sbf, idx_kj,idx_ji)
        
        edge_attr = self.interaction_blocks3(edge_attr, sbf, idx_kj,idx_ji)

        x = self.agg_2(x, edge_index, edge_attr)
        x = self.agg_3(x, edge_index)

        x = F.silu(x)
        x = self.mlp(x)
        return x, edge_attr
