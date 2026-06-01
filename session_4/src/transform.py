import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from metadata import TARGET_COLUMN, FEATURE_COLUMNS


def transform_data(df):

    # one hot encode categorical variables
    encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    encoded = encoder.fit_transform(df[["Geography", "Gender"]])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(["Geography", "Gender"])
    )

    # numerical variables
    numerical_df = df[
        [
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary"
        ]
    ].reset_index(drop=True)

    # combine data
    X = pd.concat(
        [numerical_df, encoded_df],
        axis=1
    )

    # add missing columns if necessary
    for column in FEATURE_COLUMNS:
        if column not in X.columns:
            X[column] = 0

    # keep expected columns
    X = X[FEATURE_COLUMNS]

    # target
    y = df[TARGET_COLUMN]

    return X, y