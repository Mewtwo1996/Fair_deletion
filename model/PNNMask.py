import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MLPA(torch.nn.Module):

    def __init__(self, in_feats, dim_h, dim_z):
        super(MLPA, self).__init__()
        
        self.gcn_mean = torch.nn.Sequential(
                torch.nn.Linear(in_feats, dim_h),
                torch.nn.ReLU(inplace=False),
                torch.nn.Linear(dim_h, dim_z)
                )

    def forward(self, hidden):
        # GCN encoder
        Z = self.gcn_mean(hidden)
        # inner product decoder
        adj_logits = Z @ Z.T
        return adj_logits

class PGNNMask(torch.nn.Module):
    def __init__(self, features, n_hidden=64, temperature=1) -> None:
        super(PGNNMask,self).__init__()
        #self.g_encoder = GCN_Body(in_feats = features.shape[1], n_hidden = n_hidden, out_feats = n_hidden, dropout = 0.1, nlayer = 1)
        self.Aaug = MLPA(in_feats = n_hidden, dim_h = n_hidden, dim_z =features.shape[1])
        self.temperature = temperature
        
    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs=gate_inputs.cuda()
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph
    
    def normalize_adj(self,adj):
        adj.fill_diagonal_(1)
        # normalize adj with A = D^{-1/2} @ A @ D^{-1/2}
        D_norm = torch.diag(torch.pow(adj.sum(1), -0.5)).cuda()
        adj = D_norm @ adj @ D_norm
        return adj

    def forward(self, h, alpha = 0.5, adj_orig = None):
        #h = self.g_encoder(adj, x)

        # Edge perturbation
        adj_logits = self.Aaug(h)
        ## sample a new adj
        edge_probs = torch.sigmoid(adj_logits)

        if (adj_orig is not None) :
            edge_probs = alpha*edge_probs + (1-alpha)*adj_orig

        # sampling 
        adj_sampled =self._sample_graph(adj_logits)
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        adj_sampled = self.normalize_adj(adj_sampled)


        return adj_sampled, adj_logits