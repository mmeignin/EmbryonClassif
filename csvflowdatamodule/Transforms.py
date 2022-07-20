from torchvision import transforms
import torch
from argparse import ArgumentParser
from ipdb import set_trace
import numpy as np

class TransformsComposer():
    """
    Compose and setup the transforms depending command line arguments.
    Define a series of transforms, each transform takes a dictionnary
    containing a subset of keys from ['Flow', 'Image', 'GtMask'] and
    has to return the same dictionnary with content elements transformed.
    """
    def __init__(self, augmentation) :
        transfs = []
        self.augmentations = TrAugmentVideo(augmentation)
        transfs.append(self.augmentations)
        self.TrCompose = transforms.Compose(transfs)

    def __call__(self, ret) :
        return self.TrCompose(ret)

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = TrAugmentVideo.add_specific_args(parser)
        return parser


class TrAugmentVideo() :
    """
    Data augmentation techniques for videos

    Args :
        Name augmentation (list str) : data augmentation to return
    """
    def __init__(self, augmentation) :
        self.augs = []
        augs_names =  augmentation.split('_')
        for name in augs_names :
            self.interpret_name(name)
        self.declare()

    def interpret_name(self, name) :
        if 'randombrightness' == name :
            self.augs.append(self.randombrightness)
        if 'hflip' == name :
            self.augs.append(self.hflip)
        elif (name == 'none') or (name=='') :
            pass
        else :
            raise Exception(f'Flow augmentation {name} is unknown')

    def __call__(self, ret) :
        """
        Call all augmentations defined in the init
        """
        for aug in self.augs :
            ret = aug(ret)
        return ret

    def declare(self):
         print(f'Flow Transformations : {[aug for aug in self.augs]}')

    @staticmethod
    def randombrightness(ret) :
        """
        Apply a random brightness adjust to all vectors of the flow fields
        Args :
          ret : dictionnary containing at least "Video"
        Return :
          ret dictionnary with Flow transform for one frame
              'Flow' : (Nframes, Channels ,W, H)
        """
        fct = torch.randn(1) + 1
        ret['Video'] += 0.5
        ret['Video'] = transforms.functional.adjust_brightness(ret['Video'], fct)
        ret['Video'] -= 0.5
        return ret

    @staticmethod
    def hflip(ret) :
        """
        Horizontal Flip
        Args :
          ret : dictionnary containing at least "Video"
        Return :
          ret dictionnary with Flow transform for one frame
              'Flow' : (Nframes, Channels ,W, H)
        """
        ret['Video'] = transforms.functional.hflip(ret['Video'])
        return ret

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--augmentation', type=str, default='none')
        return parser
