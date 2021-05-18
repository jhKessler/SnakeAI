import torch.nn as nn

class QNetwork(nn.Module):
    """Neural Network for playing Snake"""


    def __init__(self, inp_dim: int, outp_dim: int, nodes_per_layer: int):
        """Creates an instance of a Deep QNetwork"""
        super().__init__()
        self.inp_dim = inp_dim
        self.nodes_per_layer = nodes_per_layer

        # activation functions
        self.leaky = nn.LeakyReLU(0.2)

        self.layers = nn.ModuleList([

            nn.Linear(in_features=inp_dim*inp_dim, out_features=self.nodes_per_layer),
            nn.BatchNorm1d(num_features=self.nodes_per_layer),
            self.leaky,

            nn.Linear(in_features=self.nodes_per_layer, out_features=self.nodes_per_layer),
            nn.BatchNorm1d(num_features=self.nodes_per_layer),
            self.leaky,

            nn.Linear(in_features=self.nodes_per_layer, out_features=self.nodes_per_layer),
            nn.BatchNorm1d(num_features=self.nodes_per_layer),
            self.leaky,

            nn.Linear(in_features=self.nodes_per_layer, out_features=self.nodes_per_layer),
            nn.BatchNorm1d(num_features=self.nodes_per_layer),
            self.leaky,
            
            nn.Linear(in_features=self.nodes_per_layer, out_features=outp_dim),
            ])


    def forward(self, x):
        """Predicts action from input"""
        x = x.view(-1, self.inp_dim*self.inp_dim)
        for lyr in self.layers:
            x = lyr(x)
        return x
        