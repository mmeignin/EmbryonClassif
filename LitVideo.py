from Models.heads.Lstm import Lstm
import pytorch_lightning as pl
from einops import rearrange
from Models.VideoModels.SlowFast import SlowFast



from argparse import ArgumentParser

class LitVideo(pl.LightningModule) :
    def __init__(self,video_model, **kwargs) :
        super().__init__()
        self.model = self.init_model(video_model,**kwargs)

    def init_model(self, video_model,**kwargs) :
        if video_model == 'SlowFast' :
            return SlowFast(**kwargs)
        else :
            print(f'VideoModel {video_model} not available')

        
    def forward(self, batch) :
        """Compute feature for each input frame.

        Parameters
        ----------
        batch : 

        Returns
        -------
        batch : add field 'Features' ( Nvideos, 8)
        """
        output = self.model(batch['Video']) # ( Nvideos, 8)
        return output


    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--video_model', type=str, choices=['SlowFast'], default='SlowFast')
        parser.add_argument('--pretrained_vm', '-p_vm', action='store_true', help='Use pretrained Video_model')
        return parser
