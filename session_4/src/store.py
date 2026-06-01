import os
import joblib

from datetime import datetime

from metadata import MODEL_FOLDER, MODEL_NAME


def save_model(model):

    # create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # create filename
    filename = f"class_model-{MODEL_NAME}-{timestamp}.joblib"

    # complete path
    filepath = os.path.join(MODEL_FOLDER, filename)

    # save model
    joblib.dump(model, filepath)

    return filepath
