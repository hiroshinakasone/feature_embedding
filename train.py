import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn import CosineEmbeddingLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from embedding.dataset import UserDataset
from embedding.model import UserModel

EPOCH = 20
BATCH_SIZE = 32

def main():
    device = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")

    dataset = UserDataset("data/ml-latest-small/ratings.csv")
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.5, random_state=42, stratify=dataset.labels())
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_dataset = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

    model = UserModel(610, 128).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = CosineEmbeddingLoss(reduction="mean").to(device)

    train_losses = []
    test_losses = []
    model.train()
    for _ in tqdm(range(EPOCH)):
        _train_losses = []
        for i, (X1, X2, y) in enumerate(train_loader):
            emb1, emb2 = model(X1.to(device), X2.to(device))
            loss = criterion(emb1.to(device), emb2.to(device), y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _train_losses.append(loss.item())
            if i % 100 == 0:
                train_losses.append(np.mean(_train_losses))
                _train_losses = []

        _test_losses = []
        for i, (X1, X2, y) in enumerate(test_dataset):
            emb1, emb2 = model(X1.to(device), X2.to(device))
            loss = criterion(emb1.to(device), emb2.to(device), y.to(device))
            _test_losses.append(loss.item())
            if i % 100 == 0:
                test_losses.append(np.mean(_test_losses))
                _test_losses = []

    plt.figure()
    plt.plot(train_losses, label="Train Data")
    plt.plot(test_losses, label="Test Data")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()