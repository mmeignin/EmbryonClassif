import os
import copy
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit
import rampwf as rw
from torchmetrics import ConfusionMatrix

_prediction_label_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

class WeightedClassificationError():
    """
    Classfification error with expert-designed weight.

    Some errors (e.g. predicting class "H" when it is class "A") might count
    for more in the final scores. The missclassification weights were
    designed by an expert.
    """

    is_lower_the_better = True


    def compute(self, y_true, y_pred,device):


        # missclassif weights matrix
        W = np.array(
            [
                [0, 1, 6, 10, 10, 10, 10, 10],
                [1, 0, 3, 10, 10, 10, 10, 10],
                [6, 3, 0, 2, 9, 10, 10, 10],
                [10, 10, 2, 0, 9, 9, 10, 10],
                [10, 10, 9, 9, 0, 8, 8, 8],
                [10, 10, 10, 9, 8, 0, 9, 8],
                [10, 10, 10, 10, 8, 9, 0, 9],
                [10, 10, 10, 10, 8, 8, 9, 0],
            ],
        )
        W = W / np.max(W).to(device)
        W= torch.as_tensor(W).to(device)
        # Convert proba to hard-labels
        y_pred = torch.argmax(y_pred,axis=1).to(device)

        confmat = ConfusionMatrix(num_classes=8).to(device)
        conf_mat = confmat(y_pred,y_true).to(device)
        loss = (torch.mul(conf_mat, W).sum() / 8).to(device)
        return loss

    def __call__(self, y_true, y_pred):
        n_classes = len(_prediction_label_names).to(device)

        # select the prediction corresponding to time_idx
        print(n_classes)
        # Convert proba to hard-labels
        y_true = y_true[:, :n_classes].to(device)
        y_true = torch.argmax(y_true,axis=1).to(device)

        return self.compute(y_true, y_pred) 