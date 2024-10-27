import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset


class UserDataset(Dataset):
    def __init__(self, rating_file_path):
        self._le_movies = preprocessing.LabelEncoder()
        self._le_users = preprocessing.LabelEncoder()

        df = pd.read_csv(rating_file_path, usecols=["movieId", "userId", "rating"], engine="python")
        df.loc[df.loc[:, "rating"] < 4.0, "rating"] = -1.0
        df.loc[df.loc[:, "rating"] >= 4.0, "rating"] = 1.0
        movies = self._le_movies.fit_transform(df.loc[:, "movieId"].values)
        users = self._le_users.fit_transform(df.loc[:, "userId"].values)
        ratings = df.loc[:, "rating"].values

        size = (len(self._le_movies.classes_), len(self._le_users.classes_))
        feature_matrix = torch.sparse_coo_tensor(
            np.array((movies, users)),
            torch.tensor(ratings, dtype=torch.float32),
            size=size,
        ).to_dense()

        self._m1 = feature_matrix[0:feature_matrix.shape[0]//2]
        self._m2 = feature_matrix[feature_matrix.shape[0]//2:]
        cosine_sim = cosine_similarity(self._m1, self._m2)
        self._m1_indices, self._m2_indices = np.where((cosine_sim >= 0.8) | (cosine_sim <= -0.8))
        self._labels = np.array([1 if cosine_sim[tuple(idx)] >= 0.8 else -1 for idx in zip(self._m1_indices, self._m2_indices)], dtype=np.int8)

    def __len__(self):
        return len(self._m1_indices)

    def __getitem__(self, idx):
        m1_index = self._m1_indices[idx]
        m2_index = self._m2_indices[idx]
        return self._m1[m1_index], self._m2[m2_index], self._labels[idx]

    def labels(self):
        return self._labels
