import pytorch_lightning as pl
import torch
from einops import rearrange
from Models.backbones.SimpleConv import SimpleConv
from Models.backbones.ResNet18 import ResNet18
from Models.backbones.ResNet34 import ResNet34
from Models.backbones.Vgg11 import Vgg11
from Models.backbones.Vit import Vit

from argparse import ArgumentParser


class LitBackbone(pl.LightningModule) :
    def __init__(self, backbone,embedding,**kwargs) :
        super().__init__()
        self.model = self.init_model(backbone, **kwargs)
        self.embedding = False
        print(f'Using embedding : {self.embedding}')


    def init_model(self, backbone, **kwargs) :
        if backbone == 'SimpleConv' :
            return SimpleConv()
        elif backbone == 'ResNet18':
            return ResNet18(**kwargs)
        elif backbone == 'ResNet34':
            return ResNet34(**kwargs)
        elif backbone == 'Vgg11':
            return Vgg11(**kwargs)
        elif backbone == 'Vit' :
            return Vit(**kwargs)
        else :
            print(f'Backbone {backbone} not available')

    def forward(self, batch) :
        """Compute feature for each input frame.
        Parameters
        ----------
        batch : Dict containing
            'Video' (Nvideos, Nframes, channels, w, H)

        Returns
        -------
        batch : add field 'Features' ( Nvideos, NFrames, Nfeats)
        """
    
        b, f, c , w, h = batch['Video'].shape
        input =  rearrange(batch['Video'], 'b f c w h -> (b f) c w h') # (Nvideos*NFrames, channels, w, h)
        features = self.model(input.to(batch['Class'].device)  ) # (Nvideos*NFrames, Nfeats)
        vid_feats = rearrange(features, ' (b f) s -> b f s', b=b , f=f) 
        if self.embedding =='True' :
            b_size,frames = batch['t0'].shape
            for i in range(0,frames):
                batch['t0'][:,i] = batch['t0'][:,0]+0.15*i* 300/frames # changer pas coder en dur dans la boucle
            output = (vid_feats + batch['t0'].reshape(b_size,frames,1).to(torch.float))
        else :
            output = vid_feats
        return output.to(batch['Class'].device)  

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--backbone', '-bb', type=str, choices=['SimpleConv', 'ResNet18', 'ResNet34', 'Vgg11','Vit'], default='ResNet18')
        parser.add_argument('--pretrained_backbone', '-pb', action='store_true', help='Use pretrained backbone')
        parser.add_argument('--embedding', "-emb", type=str ,choices =['True','False'],default ='False')
        return parser

    def get_output_feats(self, img_size) :
        return self.model.get_output_feats(img_size)
