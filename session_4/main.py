from src.source import load_data
from src.transform import transform_data
from src.train import train_model
from src.store import save_model


def main():

    # load data
    df = load_data()

    # transform data
    X, y = transform_data(df)

    # train model
    model = train_model(X, y)

    # save model
    filepath = save_model(model)

    print(f"Model saved in: {filepath}")


if __name__ == "__main__":
    main()