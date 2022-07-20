import torch.nn as nn
from einops import rearrange

class Gru(nn.Module):
    def __init__(self, input_feats):
        super().__init__()
        self.input_feats = input_feats
        self.gru = nn.GRU(input_feats, 8, batch_first=True) # 4096 for (125,125)

    def forward(self, x):
        """Extract features from image

        Parameters
        ----------
        x : tensor (Nvideos, Nfeats, NFrames)

        Returns
        -------
        output : tensor (Nvideos, 8)
        """
        Nvideos = x.shape[0]
        x =  rearrange(x, 'videos features frames -> videos frames features', features=self.input_feats) #(Nvideos, Nfeats, NFrames)
        x, h = self.gru(x) # x : (Nvideos, NFrames, 8)
        x = x[:, -1, :] # (Nvideos, 8)
        assert x.shape==(Nvideos, 8), f'Dimension output should be ({Nvideos},{8}) and is ({x.shape})'
        return x
