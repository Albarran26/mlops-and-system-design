DATASET_PATH = "datasets/Churn_Modelling_train_test.csv"

TARGET_COLUMN = "Exited"

MODEL_NAME = "Daniel"

MODEL_PARAMS = {"max_depth": 5, "random_state": 42}

FEATURE_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
    "Gender_Male",
]

MODEL_FOLDER = "session_4/models/"
