import numpy as np
import torch

def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(X.transpose(0, 1))
    return torch.mm(X, U[:, :k])