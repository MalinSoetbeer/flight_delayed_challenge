import numpy as np
import pandas as pd


def transform_altitude(df: pd.DataFrame) -> pd.DataFrame:
    df["altitude_mean_log"] = np.log(df["altitude_mean_meters"])
    df = df.drop(
        [
            "altitude_mean_meters",
        ],
        axis=1,
    )
    return df


#'Unnamed: 0' and Quakers
def drop_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    df = df.drop([col_name], axis=1)
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    altitude_low_meters_mean = 1500.3684210526317
    altitude_high_meters_mean = 1505.6315789473683
    altitude_mean_log_mean = 7.0571530664031155
    df["altitude_low_meters"] = df["altitude_low_meters"].fillna(
        altitude_low_meters_mean
    )
    df["altitude_high_meters"] = df["altitude_high_meters"].fillna(
        altitude_high_meters_mean
    )
    df["altitude_mean_log"] = df["altitude_mean_log"].fillna(altitude_mean_log_mean)
    return df


def time_data(df):
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    df["flight_year"] = df["flight_date"].dt.year
    df["flight_month"] = df["flight_date"].dt.month_name()
    df["flight_day"] = df["flight_date"].dt.day
    df["flight_dayofWeek"] = df["flight_date"].dt.day_name()
    df["flight_dayofYear"] = df["flight_date"].dt.dayofyear
    df["flight_weekofYear"] = df["flight_date"].dt.isocalendar().week.astype(int)

    df["scheduled_time_departure"] = pd.to_datetime(df["scheduled_time_departure"])
    df["scheduled_time_arrival"] = pd.to_datetime(
        df["scheduled_time_arrival"], format="%Y-%m-%d %H.%M.%S"
    )
    df["departure_Hour"] = df["scheduled_time_departure"].dt.hour
    df["departure_Minutes"] = df["scheduled_time_departure"].dt.minute


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
