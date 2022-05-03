import sys
import pandas as pd
import pickle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    accuracy_score,
    f1_score,
    classification_report,
)
import warnings

warnings.filterwarnings("ignore")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        square=True,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["On time", "Delayed"],
        yticklabels=["On time", "Delayed"],
    )
    plt.xlabel("predicted label")
    plt.ylabel("actual label")
    plt.show()


def error_analysis(y_test, y_pred_test):
    """Generated true vs. predicted values and residual scatter plot for models
    Args:
        y_test (array): true values for y_test
        y_pred_test (array): predicted values of model for y_test
    """
    y_pred_test = np.array(y_pred_test)
    y_test = np.array(y_test)
    # Calculate residuals
    residuals = y_test - y_pred_test
    # Plot real vs. predicted values
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    plt.subplots_adjust(right=1)
    plt.suptitle("Error Analysis")
    ax[0].scatter(y_pred_test, y_test, color="#FF5A36", alpha=0.7)
    ax[0].plot([-400, 350], [-400, 350], color="#193251")
    ax[0].set_title("True vs. predicted values", fontsize=16)
    ax[0].set_xlabel("predicted values")
    ax[0].set_ylabel("true values")
    ax[0].set_xlim((y_pred_test.min() - 10), (y_pred_test.max() + 10))
    ax[0].set_ylim((y_test.min() - 40), (y_test.max() + 40))
    ax[1].scatter(y_pred_test, residuals, color="#FF5A36", alpha=0.7)
    ax[1].plot([-400, 350], [0, 0], color="#193251")
    ax[1].set_title("Residual Scatter Plot", fontsize=16)
    ax[1].set_xlabel("predicted values")
    ax[1].set_ylabel("residuals")
    ax[1].set_xlim((y_pred_test.min() - 10), (y_pred_test.max() + 10))
    ax[1].set_ylim((residuals.min() - 10), (residuals.max() + 10))
    plt.show()
