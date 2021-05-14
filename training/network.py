import torch.nn as nn

class QNetwork(nn.Module):
    """Neural Network for playing Snake"""


    def __init__(self, inp_dim: int, outp_dim: int):
        """Creates an instance of a Deep QNetwork"""
        super().__init__()
        self.inp_dim = inp_dim

        # activation functions
        self.leaky = nn.LeakyReLU(0.2)
        self.sig = nn.Sigmoid()

        self.layers = nn.ModuleList([
            nn.Linear(in_features=inp_dim*inp_dim, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            self.leaky,
            nn.Linear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            self.leaky,
            nn.Linear(in_features=1024, out_features=outp_dim),
            self.sig
            ])


    def forward(self, x):
        """Predicts action from input"""
        x = x.view(-1, self.inp_dim*self.inp_dim)
        for lyr in self.layers:
            x = lyr(x)
        return x
        