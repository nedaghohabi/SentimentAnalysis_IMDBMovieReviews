from utils.timeit import timeit
from runners.epoch_runner import EpochRunner
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn


@timeit
def train_ml_model(model, X_train, y_train):
    """Trains the model."""
    model.fit(X_train, y_train)
    return model


@timeit
def evaluate_ml_model(model, X, y, dataset_name):
    """Evaluates the model and logs results."""
    preds = model.predict(X)
    return preds