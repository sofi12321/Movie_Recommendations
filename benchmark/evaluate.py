from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error


def set_seed(seed):
    """
    Manual seeding
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(model, df, device):
    """
    Make a prediction by a model
    :param model: model to be used
    :param df: input data
    :param device: where to run a model
    :return: predictions to each datapoint in df
    """
    model = model.to(device)
    X = torch.tensor(df.values).float().to(device)
    model.eval()
    with torch.no_grad():
        y_pred = model(X).reshape(-1).cpu().detach().numpy()
    return y_pred


def evaluate_ratings_recommendations(pred, true_y):
    """
    Print evaluation results of MAE, MSE and RMSE
    :param pred: predicted ratings
    :param true_y: true ratings
    """
    print("MAE:", mean_absolute_error(true_y, pred))
    print("MSE:", mean_squared_error(true_y, pred, squared=True))
    print("RMSE:", mean_squared_error(true_y, pred, squared=False))


def recommend_movies(user_id, ua_base, ua_test, train_dataset_columns, device, k=5):
    """
    Print top k movies that are most relevant to a user with user_id.
    Predict movies unseen by user before both in train and test data

    :param user_id: user id (from a users dataset)
    :param k: maximal number of movies to output
    :param ua_base: train dataset, used to avoid prediction of already seen movie
    :param ua_test: validation dataset, used to avoid prediction of already seen movie
    :param train_dataset_columns: used to sort columns in a right way to model input
    :param device: where to run a model
    """
    watched_movies = ua_base[ua_base["user_id"] == user_id]["item_id"].to_numpy()
    watched_movies_1 = ua_test[ua_test["user_id"] == user_id]["item_id"].to_numpy()
    watched_movies = np.unique(np.concatenate((watched_movies, watched_movies_1), 0))

    # Prepare a data to put in a model
    df = processed_movies.drop(watched_movies, axis=0)
    df["item_id"] = df.index.copy()
    df["user_id"] = user_id
    df = df.merge(processed_users, left_on="user_id", right_index=True)
    df = df.drop("user_id", axis=1).reset_index(drop=True)

    # Predict movies ratings
    preds = predict(model, df[train_dataset_columns], device)
    # Take ids of the top-rated movies
    res = df.loc[np.argsort(-preds)]["item_id"].values[:k]

    print(f"Recommended movies for user {user_id} are")
    for ind in res:
        print(movies.loc[ind, "movie_title"])


def get_model(model_path):
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=426, out_features=150),
        torch.nn.LeakyReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(in_features=150, out_features=20),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(in_features=20, out_features=1),
        torch.nn.Sigmoid()
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


if __name__ == "__main__":
    set_seed(21)

    # Load pretrained model
    model = get_model("models/model.pt")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Preprocessed dataset on which calculate metrics
    # Used ua_test from MovieLens 100K
    # Ratings are integers in range [0, 4]
    dataset = pd.read_csv("benchmark/data/dataset.csv", index_col=0)

    pred = predict(model, dataset.drop("rating", axis=1).loc[1:3], device)

    print("Evaluation of initial ratings:")
    evaluate_ratings_recomendations(pred * 4 + 1, dataset.loc[1:3, "rating"].values + 1)
    print()
    print("Evaluation of normalized ratings:")
    evaluate_ratings_recomendations(pred, dataset.loc[1:3, "rating"].values / 4)
    print()

    # Initial train dataset, used to avoid prediction of already seen movie
    ua_base = pd.read_csv("ml-100k/ua.base", sep="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])
    # Initial validation dataset, used to avoid prediction of already seen movie
    ua_test = pd.read_csv("ml-100k/ua.test", sep="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])
    # Processed dataset columns, used to sort columns in a right way to model input
    train_dataset_columns = pd.read_csv("train_dataset.csv", index_col=0).drop("rating", axis=1).columns

    recommend_movies(21, ua_base, ua_test, train_dataset_columns, device, k=10)
