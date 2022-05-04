import numpy as np
import pandas as pd
from geopy.distance import geodesic


def fix_airport(df: pd.DataFrame) -> pd.DataFrame:
    """Berlin Schönefeld is referenced as SXF in the zindi data, but only
    exists as BER in the airports data. Therefore we replace SXF with BER
    Args:
        df (pd.DataFrame): dataframe containing the zindi data
    Returns:
        pd.DataFrame: dataframe with replaced airports
    """
    df["DEPSTN"] = df["DEPSTN"].str.replace("SXF", "BER")
    df["ARRSTN"] = df["ARRSTN"].str.replace("SXF", "BER")
    return df


def lat_lon_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the flight distance from the latitude and longitude data
    Args:
        df (pd.DataFrame): dataframe containing the zindi data and the merged airports data
    Returns:
        pd.DataFrame: df with distance column
    """
    point1 = (df.lat_DEP.values, df.lon_DEP.values)
    point2 = (df.lat_ARR.values, df.lon_ARR.values)

    dist = []
    for num in range(len(point1[0])):
        pt1 = point1[0][num], point1[1][num]
        pt2 = point2[0][num], point2[1][num]
        dist.append(geodesic(pt1, pt2).km)
    df["distance"] = dist
    return df


def create_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add custom features to the dataset
    Args:
        df (pd.DataFrame): dataframe containing the zindi data and the merged airports data

    Returns:
        pd.DataFrame: final dataframe
    """
    df["DATOP"] = pd.to_datetime(df["DATOP"])
    df["STD"] = pd.to_datetime(df["STD"])
    df["STA"] = pd.to_datetime(df["STA"], format="%Y-%m-%d %H.%M.%S")
    df["delay_or_onTime"] = df["target"].apply(
        lambda x: "on_time" if x <= 10.0 else "delay"
    )
    df["delayed"] = df["target"].apply(lambda x: 0 if x <= 10.0 else 1)
    df["domestic"] = (df.country_DEP == df.country_ARR).astype("int")
    df["dep_hour"] = df["STD"].dt.hour
    df["dep_weekday"] = df.STD.dt.day_name()
    df["duration_min"] = (df.STA - df.STD).dt.total_seconds() / 60
    df["arr_hour"] = df["STA"].dt.hour
    df["flight_month"] = df["DATOP"].dt.month
    df["flight_month_name"] = df["DATOP"].dt.month_name()
    df["year"] = df["STD"].dt.year
    return df


# Function to see how many NaN values there are in the column and how many rows have the entry 0 in these columns.  It should help to facilitate the action of deleting them or filling them with zero if necessary.
def missing_values_table(df):
    # count all zero values of the df in a list
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    # count all null values in a list
    mis_val = df.isnull().sum()
    # calculates the perc of the null values in the df an put the values in a list
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # put the list in a df and concate them
    mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)

    # renames the columns
    mz_table = mz_table.rename(
        columns={0: "Zero Values", 1: "Missing Values", 2: "% of Total Values"}
    )

    # add two more columns for a better inside
    mz_table["Total Zero and Missing Values"] = (
        mz_table["Zero Values"] + mz_table["Missing Values"]
    )
    mz_table["Data Type"] = df.dtypes

    mz_table = (
        mz_table[mz_table.iloc[:, 1] != 0]
        .sort_values("% of Total Values", ascending=False)
        .round(1)
    )

    print(
        "Your selected dataframe has "
        + str(df.shape[1])
        + " columns and "
        + str(df.shape[0])
        + " Rows.\n"
        "There are " + str(mz_table.shape[0]) + " columns that have missing values."
    )

    return mz_table


def drop_column(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """Drop specified columns from dataframe
    Args:
        df (pd.DataFrame): dataframe containing the zindi data
        cols_to_drop (list): list of strings identifying the columns to drop
    Returns:
        pd.DataFrame: dataframe with dropped columns
    """

    df2 = df.drop(cols_to_drop, axis=1)
    return df2


def drop_rows(df):

    df3 = df.drop(df[df.DEPSTN == df.ARRSTN].index)

    return df3


def reset_indices(*args):
    for i in args:
        i.reset_index(drop=True, inplace=True)


def OneHotEncoder_labels(X_train, columns):
    one_hot_class = pd.get_dummies(X_train, columns=columns, drop_first=True)
    one_hot_class = list(one_hot_class.columns)
    return one_hot_class


def Encode_categorical_features(X_train, encoder, columns):

    # Fit OneHotEncoder to X, then transform X. Here Train Data
    X_train_dummie_columns = pd.DataFrame(encoder.fit_transform(X_train[columns]))
    X_train_class = X_train.drop(columns, axis=1)
    X_train_class = X_train.join(X_train_dummie_columns)

    return X_train_class
