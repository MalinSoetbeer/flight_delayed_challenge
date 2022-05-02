import numpy as np
import pandas as pd


def dummie_class(X):
    """Returns naive predictions based on the flight being domestic or not

    Args:
        X (DataFrame): Dataframe to be classified

    Returns:
        Series : Classifications. 1 = delayed, 0 = on time
    """
    y_pred = X["domestic"].apply(lambda x: 0 if x == 1 else 1)
    return y_pred


def dummie_reg(X):

    y_pred = [89.38 for _ in range(len(X))]
    return y_pred
