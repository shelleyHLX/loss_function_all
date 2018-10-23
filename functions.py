# coding: utf-8


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mse(true, pred):
    """
    true: array of true values
    pred: array of predicted values

    returns: mean square error loss
    """

    return np.sum((true - pred)**2)

