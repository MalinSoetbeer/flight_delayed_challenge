import pickle
import sys
import warnings
from re import X

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
RSEED = 42
# Import airport data
import airportsdata

airports = airportsdata.load()

# Import scripts
from feature_engineering import *
from prepare_flight_data import *

flight_data = sys.argv[1]


# Loading the data in and Shape to fit
airports = pd.DataFrame(airports).T.reset_index(drop=True)
# test data
train_df = pd.read_csv(flight_data)

print("Feature engineering on train")
train_df = fix_airport(train_df)
train_airport_df = merge_airports(train_df, airports)

# create all featrues for firstly more insights
train_airport_df = create_feature(train_airport_df)
train_airport_df = lat_lon_distance(train_airport_df)

# List of dropping feature for classification
drop_features_class = [
    "DATOP",
    "ID",
    "FLTID",
    "STD",
    "STA",
    "target",
    "icao_DEP",
    "iata_DEP",
    "name_DEP",
    "city_DEP",
    "subd_DEP",
    "country_DEP",
    "elevation_DEP",
    "lat_DEP",
    "lon_DEP",
    "tz_DEP",
    "icao_ARR",
    "iata_ARR",
    "name_ARR",
    "city_ARR",
    "subd_ARR",
    "country_ARR",
    "elevation_ARR",
    "lat_ARR",
    "lon_ARR",
    "tz_ARR",
    "arr_hour",
    "flight_month",
    "delay_or_onTime",
    "delayed",
]


# Feature and target variable for Classification modelling
train_airport_df = drop_rows(train_airport_df)
X_class = drop_column(train_airport_df, drop_features_class)
y_class = train_airport_df["delayed"]
print("==================")
print("Training data for classification: {}".format(X_class.shape[0]))
print("Test data for classification: {}".format(y_class.shape[0]))
# Split the 'features' and 'target' data into training and testing sets for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, stratify=y_class, random_state=RSEED
)

# resetting indices
reset_indices(X_train_class, X_test_class)

# Creating list for categorical predictors/features
# (dates are also objects so if you have them in your data you would deal with them first)
cat_features = [
    "DEPSTN",
    "ARRSTN",
    "STATUS",
    "AC",
    "domestic",
    "dep_hour",
    "dep_weekday",
    "flight_month_name",
    "year",
]

num_features = ["duration_min", "distance"]

# Initialize a scaler, then apply it to the features
scaler_class = StandardScaler()
scaler_class.fit(X_train_class[num_features])
X_train_class[num_features] = scaler_class.transform(X_train_class[num_features])
X_test_class[num_features] = scaler_class.transform(X_test_class[num_features])

# Rename the columns from the Encoded df
col = OneHotEncoder_labels(X_train_class, cat_features)

# One-hot encode the 'features' data using sklearn OneHotEncoder to Encode categorical features as a one-hot numeric array.
encoder_class = OneHotEncoder(handle_unknown="ignore", sparse=False, drop="first")
encoder_class.fit(X_train_class[cat_features])

# Fit OneHotEncoder to X, then transform X. Here Train Data
X_train_dummie_columns = pd.DataFrame(
    encoder_class.transform(X_train_class[cat_features])
)
X_train_class = X_train_class.drop(cat_features, axis=1)
X_train_class = X_train_class.join(X_train_dummie_columns)
X_train_class.columns = col

# Fit OneHotEncoder to X, then transform X. Here Test Data
X_test_dummie_columns = pd.DataFrame(
    encoder_class.transform(X_test_class[cat_features])
)
X_test_class = X_test_class.drop(cat_features, axis=1)
X_test_class = X_test_class.join(X_test_dummie_columns)
X_test_class.columns = col

## in order to exemplify how the predict will work.. we will save the y_train
print("Saving test data in the data folder")
X_test_class.to_csv("data/X_test.csv", index=False)
y_test_class.to_csv("data/y_test.csv", index=False)

# model
print("Training a Support Vector Classifier")
# Initialize the SVC classifier
SVC_clf = SVC(kernel="rbf", cache_size=1000, C=10, gamma=0.06).fit(
    X_train_class, y_train_class
)
y_train__class_pred = SVC_clf.predict(X_train_class)
f1_scores = f1_score(y_train_class, y_train__class_pred, average="macro")

y_test_pred_class = SVC_clf.predict(X_test_class)
f1_scores_test = f1_score(y_test_class, y_test_pred_class, average="macro")


print(f"F1 macro on train is: {f1_scores}")
print(f"F1 macro on test is: {f1_scores_test}")
print("Model training completed")
# saving the model
print("Saving model in the model folder")
filename = "models/SVC_model.sav"
pickle.dump(SVC_clf, open(filename, "wb"))
