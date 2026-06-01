import pandas as pd

from src.transform import transform_data


def test_transform_data_returns_expected_shape():

    df = pd.DataFrame(
        {
            "CreditScore": [600, 700],
            "Geography": ["France", "Germany"],
            "Gender": ["Female", "Male"],
            "Age": [35, 42],
            "Tenure": [3, 5],
            "Balance": [10000.0, 20000.0],
            "NumOfProducts": [1, 2],
            "HasCrCard": [1, 0],
            "IsActiveMember": [1, 0],
            "EstimatedSalary": [50000.0, 60000.0],
            "Exited": [0, 1],
        }
    )

    X, y = transform_data(df)

    assert X.shape[0] == 2
    assert y.shape[0] == 2


def test_transform_data_removes_target_from_features():

    df = pd.DataFrame(
        {
            "CreditScore": [600, 700],
            "Geography": ["France", "Germany"],
            "Gender": ["Female", "Male"],
            "Age": [35, 42],
            "Tenure": [3, 5],
            "Balance": [10000.0, 20000.0],
            "NumOfProducts": [1, 2],
            "HasCrCard": [1, 0],
            "IsActiveMember": [1, 0],
            "EstimatedSalary": [50000.0, 60000.0],
            "Exited": [0, 1],
        }
    )

    X, y = transform_data(df)

    assert "Exited" not in X.columns
