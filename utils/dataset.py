import os
import numpy as np
import torch
from torch.utils.data import Dataset


class KGDataset(Dataset):
    def __init__(self, x_train):
        self.x_train = torch.FloatTensor(x_train.toarray())

        self.data = self.x_train

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return self.x_train[idx]


class KGTripleDataset(Dataset):
    def __init__(self, ckg_graph):
        self.triples = []
        # 遍历知识图谱中所有的边，注意 MultiDiGraph 的 edges(keys=True) 返回 (h, t, r)
        for h, t, r in ckg_graph.edges(keys=True):
            self.triples.append((h, r, t))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        return {
            'head': torch.tensor(h, dtype=torch.long),
            'relation': torch.tensor(r, dtype=torch.long),
            'tail': torch.tensor(t, dtype=torch.long)
        }

# class RecDataset(Dataset):
#     def __init__(self, train_matrix):
#         coomat = train_matrix.tocoo()
#         self.rows = coomat.row
#         self.cols = coomat.col
#         self.dokmat = train_matrix.todok()
#         self.negs = np.zeros(len(self.rows)).astype(np.int32)
#
#     def neg_sampling(self, n_item):
#         for i in range(len(self.rows)):
#             u = self.rows[i]
#             while True:
#                 i_neg = np.random.randint(n_item)
#                 if (u, i_neg) not in self.dokmat:
#                     break
#             self.negs[i] = i_neg
#
#     def __len__(self):
#         return len(self.rows)
#
#     def __getitem__(self, idx):
#         return self.rows[idx], self.cols[idx], self.negs[idx]
#         #return torch.tensor(self.rows[idx], dtype=torch.long), torch.tensor(self.cols[idx], dtype=torch.long), torch.tensor(self.negs[idx], dtype=torch.long)

class RecDataset(Dataset):
    def __init__(self, train_matrix, sample_indices=None):

        coo = train_matrix.tocoo()
        full_rows = coo.row
        full_cols = coo.col
        self.dokmat = train_matrix.todok()
        if sample_indices is not None:
            self.rows = full_rows[sample_indices]
            self.cols = full_cols[sample_indices]
        else:
            self.rows = full_rows
            self.cols = full_cols
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self, n_item):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(n_item)
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]