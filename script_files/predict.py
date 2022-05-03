import pickle
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

# Import models
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    mean_squared_error,
    r2_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

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


def classification_models():
    models_class = []

    models_class.append(("KNN", KNeighborsClassifier()))
    models_class.append(("SVC", SVC()))
    models_class.append(("LR", LogisticRegression()))
    models_class.append(("DT", DecisionTreeClassifier()))
    models_class.append(("RF", RandomForestClassifier()))
    models_class.append(("GNB", GaussianNB()))
    models_class.append(("XGB", XGBClassifier()))
    models_class.append(("ADA", AdaBoostClassifier()))
    models_class.append(("SGD", SGDClassifier()))

    return models_class


def regression_models():
    models_reg = []

    models_reg.append(("KNN", KNeighborsRegressor()))
    models_reg.append(("LREG", LinearRegression()))
    models_reg.append(("SVM", SVR()))
    models_reg.append(("DT", DecisionTreeRegressor()))
    models_reg.append(("RF", RandomForestRegressor()))
    models_reg.append(("XGBC", XGBRegressor()))
    models_reg.append(("ADA", AdaBoostRegressor()))
    models_reg.append(("SGD", SGDRegressor()))

    return models_reg

# Calculate metric
def calculate_metrics(y_test, y_pred_test):
    """Calculate and print out RMSE and R2 for train and test data

    Args:
        y_test (array): true values of y_train
        y_pred_test (array): predicted values of model for y_test
    """
    
    # Calculate metric
    print("Metrics on test data")  
    rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    # you can get the same result with this line:
    # rmse = np.sqrt(mean_squared_error(y_test,y_pred_test))

    r2 = r2_score(y_test,y_pred_test)
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 3))
    print("---"*10)