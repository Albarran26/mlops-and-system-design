from sklearn.tree import DecisionTreeClassifier

from metadata import MODEL_PARAMS


def train_model(X, y):

    model = DecisionTreeClassifier(**MODEL_PARAMS)

    model.fit(X, y)

    return model
