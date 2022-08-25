import os
import copy
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit
from torchmetrics import ConfusionMatrix

_prediction_label_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

class WeightedClassificationError(torch.nn.Module):
    n_classes = 8
    Wmax=10
    W = torch.tensor (
            [[0, 1, 6, 10, 10, 10, 10, 10],
                [1, 0, 3, 10, 10, 10, 10, 10],
                [6, 3, 0, 2, 9, 10, 10, 10],
                [10, 10, 2, 0, 9, 9, 10, 10],
                [10, 10, 9, 9, 0, 8, 8, 8],
                [10, 10, 10, 9, 8, 0, 9, 8],
                [10, 10, 10, 10, 8, 9, 0, 9],
                [10, 10, 10, 10, 8, 8, 9, 0],]
        )
    def __init__(
        self, name="WeightedClassificationError", precision=2, time_idx=0
    ):
        self.precision = precision

    def compute(self, y_pred,y_true,device):
        confmat = ConfusionMatrix(num_classes=8).to(device)
        loss = torch.sum(torch.multiply(confmat(y_pred,y_true),self.W.to(device)))/ (self.n_classes * self.Wmax)
        return loss

    def __call__(self, y_true, y_pred,device):
        y_pred = y_pred.to(device)
        y_true = y_true.to(device)
        return self.compute(y_true, y_pred,device) 