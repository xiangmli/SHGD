import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import svds
from models.denoiser import Denoiser, MLP_Denoiser
import scipy.sparse as sp



def sparse_array_to_tensor(sparse_array):
    if not hasattr(sparse_array, 'tocoo'):
        print("not hasattr!")

        sparse_array = sp.coo_matrix(sparse_array)
    else:
        print("hasattr!")
        sparse_array = sparse_array.tocoo()
    indices = np.vstack((sparse_array.row, sparse_array.col))
    indices = torch.tensor(indices, dtype=torch.long)
    values = torch.tensor(sparse_array.data, dtype=torch.float32)
    shape = sparse_array.shape
    return torch.sparse.FloatTensor(indices, values, torch.Size(shape))

def compute_eigen(adj_right, cutoff):
    _, values, vectors = svds(adj_right, k=cutoff)
    idx = np.argsort(values)[::-1]
    values = values[idx] ** 2
    vectors = vectors[idx]
    return {'cutoff': cutoff, 'values': values, 'vectors': vectors}


def compute_eigen_with_cache(adj_right, cutoff, cache_file):
    if os.path.exists(cache_file):
        eigen = np.load(cache_file, allow_pickle=True).item()
        if eigen['cutoff'] >= cutoff:
            eigen['values'] = eigen['values'][:cutoff]
            eigen['vectors'] = eigen['vectors'][:cutoff]
            return eigen
    eigen = compute_eigen(adj_right, cutoff)
    np.save(cache_file, eigen)
    return eigen




class Recommender(nn.Module):
    def __init__(self):
        super(Recommender, self).__init__()

    def reconstruct(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x, training=False)

    def recommend(self, x, k, mask):
        self.eval()
        with torch.no_grad():
            scores = self.forward(x, training=False)
            scores = scores.masked_fill(mask, float('-inf'))
            _, indices = torch.topk(scores, k, dim=1)
            return indices


class SHGD(Recommender):
    def __init__(
            self,
            interaction,
            kgsp,
            entity_embeddings,
            device,
            args,
            cache_path=None,
            embed_dim=200,
            activation='swish',
            initializer='glorot_uniform',
            dropout=0.5,
            norm_ord=1,
            T=2,
            t=None,
            alpha=2.5,
            ideal_weight=0.0,
            ideal_cutoff=200,
            noise_decay=1.0,
            noise_scale=0.0,
            ablation=None,
            **kwargs,
    ):
        super(SHGD, self).__init__()
        self.dropout = dropout
        n_items = interaction.shape[1]
        self.mlp = True

        if self.mlp:
            self.denoiser = MLP_Denoiser(n_items, [1000], T)
            print("MLP bulit!")
        else:
            self.denoiser = Denoiser(n_items, embed_dim, activation, initializer, norm_ord, T, ablation)


        user_deg = interaction.sum(axis=1)[:, np.newaxis]
        item_deg = interaction.sum(axis=0)[np.newaxis, :]
        self.device = device
        self.kgsp = kgsp
        self.n_entities = self.kgsp.shape[0]

        epsilon = 1e-10
        adj_right = np.power((user_deg+epsilon), -1 / 4) * interaction * np.power((item_deg+epsilon), -1 / 2)

        self.adj_right = sparse_array_to_tensor(adj_right.astype(np.float32)).to(device)
        self.adj_left = sparse_array_to_tensor(adj_right.T.astype(np.float32)).to(device)

        self.ideal_weight = ideal_weight
        if self.ideal_weight == 0.0:
            ideal_cutoff = 1
        print("calculating eigen。。。")
        if cache_path is None:
            eigen = compute_eigen(adj_right, ideal_cutoff)
        else:
            cache_file = os.path.join(cache_path, 'eigen.npy')
            eigen = compute_eigen_with_cache(adj_right, ideal_cutoff, cache_file)
        print("done!")
        # 将 eigen 数据注册为 buffer，保证随模型保存移动
        self.register_buffer('eigen_val', torch.tensor(eigen['values'], dtype=torch.float32))
        self.register_buffer('eigen_vec', torch.tensor(eigen['vectors'], dtype=torch.float32))
        self.entity_embeddings = entity_embeddings
        self.T = T
        self.t = np.linspace(0, T, T + 1, dtype=np.int32) if t is None else t
        self.alpha = alpha
        self.noise_decay = noise_decay
        self.noise_scale = noise_scale

        self.loss_tracker = 0.0


        self.weight_velocity = nn.Parameter(torch.empty(args.latdim, args.latdim))
        nn.init.xavier_uniform_(self.weight_velocity)
        self.mlp_second = nn.Linear(args.latdim, n_items)

    def prop(self, x):
        #device = x.device
        # adj_right = self.adj_right.to(device)
        # adj_left = self.adj_left.to(device)
        x_prop = torch.sparse.mm(self.adj_right, x.t())
        x_prop = torch.sparse.mm(self.adj_left, x_prop)
        x_prop = x_prop.t()
        return x_prop / self.eigen_val[0]

    def ideal(self, x, cutoff=None):
        eigen_vec = self.eigen_vec[:cutoff] if cutoff is not None else self.eigen_vec
        x_ideal = torch.matmul(x, eigen_vec.t())
        x_ideal = torch.matmul(x_ideal, eigen_vec)
        return x_ideal

    def smooth(self, x):
        if self.ideal_weight:
            x_smooth = self.prop(x) + self.ideal_weight * self.ideal(x)
            return x_smooth / (1 + self.ideal_weight)
        else:
            return self.prop(x)

    def filter(self, x, Ax, t):
        t = t.float()
        return x + self.alpha * t / self.T * (Ax - x)

    def sigma(self, t):
        t = t.float()
        return self.noise_scale * (self.noise_decay ** (self.T - t))

    def denoise(self, z_t, c, Ac, t, training=False):
        # 若 t 维度不足则广播至 (batch, 1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        x_pred = self.denoiser(z_t, c, Ac, t, training)
        return x_pred


    def divergence(self, batch_item_ids):
        device = self.device
        x = self.entity_embeddings
        x = x.to(device)
        #indices = self.kgsp._indices()
        # indices[0]: 源节点 i， indices[1]: 邻居节点 j
        #src, dst = indices[0], indices[1]
        #src, dst = src.to(x.device), dst.to(x.device)
        src = torch.from_numpy(self.kgsp.row.astype(np.int64)).to(device)
        dst = torch.from_numpy(self.kgsp.col.astype(np.int64)).to(device)
        delta = x[dst] - x[src]
        V = torch.tanh(torch.matmul(delta, self.weight_velocity))
        message = V * x[dst]
        second_term = torch.zeros_like(x)
        second_term = second_term.index_add(0, src, message)
        second_term = second_term[batch_item_ids]
        second_term = self.mlp_second(second_term)
        return second_term

    def forward(self, x, batch_item_ids, training=True):
        self.train()

        if training:
            batch_size = x.shape[0]
            device = x.device
            t = torch.randint(low=1, high=self.T + 1, size=(batch_size, 1), device=device, dtype=torch.int64)
            Ax = self.smooth(x)
            z_t = self.filter(x, Ax, t)
            second_term = self.divergence(batch_item_ids)
            z_t = z_t + second_term
            if self.noise_scale > 0.0:
                eps = torch.randn_like(x)
                z_t = z_t + self.sigma(t) * eps
            c = F.dropout(x, p=self.dropout, training=True)
            Ac = self.smooth(c)
            x_pred = self.denoise(z_t, c, Ac, t, training)

            loss = torch.sum((x - x_pred) ** 2, dim=1).mean()
            return loss
        else:
            Ax = self.smooth(x)
            batch_size = x.shape[0]
            device = x.device
            t_last = torch.tensor(self.t[-1],dtype=torch.int64, device=device)
            #t_last = t_last.repeat(batch_size, 1)
            z_t = self.filter(x, Ax, t_last)
            second_term = self.divergence(batch_item_ids)
            z_t = z_t + second_term
            for i in range(len(self.t) - 1, 0, -1):
                t_val = self.t[i]
                s_val = self.t[i - 1]
                t_tensor = torch.tensor(t_val, dtype=torch.int64, device=device)
                t_tensor = t_tensor.repeat(batch_size, 1)
                s_tensor = torch.tensor(s_val, dtype=torch.int64, device=device)
                x_pred = self.denoise(z_t, x, Ax, t_tensor, training=False)
                Ax_pred = self.smooth(x_pred)
                z_s_pred = self.filter(x_pred, Ax_pred, s_tensor)
                if self.noise_decay > 0.0:
                    z_t_pred = self.filter(x_pred, Ax_pred, t_tensor)
                    z_t = z_s_pred + (self.noise_decay ** (t_val - s_val)) * (z_t - z_t_pred)
                else:
                    z_t = z_s_pred
            return z_t



    def compile(self, optimizers=[None, None], learning_rates=[1e-3, 1e-3], weight_decays=[0.0, 0.0], **kwargs):
        self.optimizers = []
        optimizer0 = optimizers[0] if optimizers[0] is not None else torch.optim.Adam(
            [self.denoiser.item_embed], lr=learning_rates[0], weight_decay=weight_decays[0]
        )
        optimizer1 = optimizers[1] if optimizers[1] is not None else torch.optim.Adam(
            self.denoiser.mlp_weights, lr=learning_rates[1], weight_decay=weight_decays[1]
        )
        self.optimizers = [optimizer0, optimizer1]
