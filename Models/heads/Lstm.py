import torch.nn as nn
from einops import rearrange
<<<<<<< HEAD
from csvflowdatamodule.utils import NBClass
=======
>>>>>>> c5b426a3415e5d68823f62d2649f8da31a66a0e4

class Lstm(nn.Module):
    def __init__(self, input_feats):
        super().__init__()
        self.input_feats = input_feats
<<<<<<< HEAD
        self.lstm = nn.LSTM(input_feats,NBClass,1, batch_first=True) # 4096 for (125,125)
=======
        self.lstm = nn.LSTM(input_feats,8,5, batch_first=True) # 4096 for (125,125)
>>>>>>> c5b426a3415e5d68823f62d2649f8da31a66a0e4

    def forward(self, x):
        """Extract features from image

        Parameters
        ----------
        x : tensor (Nvideos, Nfeats, NFrames)

        Returns
        -------
<<<<<<< HEAD
        output : tensor (Nvideos, NBClass8)
        """
        Nvideos = x.shape[0]
        x =  rearrange(x, 'videos features frames -> videos frames features', features=self.input_feats) #(Nvideos, Nfeats, NFrames)
        x, h = self.lstm(x) # x : (Nvideos, NFrames, NBClass)
        x = x[:, -1, :] # (Nvideos, NBClass)
        assert x.shape==(Nvideos, NBClass), f'Dimension output should be ({Nvideos},{NBClass}) and is ({x.shape})'
=======
        output : tensor (Nvideos, 8)
        """
        Nvideos = x.shape[0]
        x =  rearrange(x, 'videos features frames -> videos frames features', features=self.input_feats) #(Nvideos, Nfeats, NFrames)
        x, h = self.lstm(x) # x : (Nvideos, NFrames, 8)
        x = x[:, -1, :] # (Nvideos, 8)
        assert x.shape==(Nvideos, 8), f'Dimension output should be ({Nvideos},{8}) and is ({x.shape})'
>>>>>>> c5b426a3415e5d68823f62d2649f8da31a66a0e4
        return x
