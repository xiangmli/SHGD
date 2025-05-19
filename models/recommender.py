import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
import scipy.sparse as sp
from models.AttnHGCN import AttnHGCN
from utils.helpers import knowledge_forward
from utils.dataset import RecDataset
from torch.utils.data import DataLoader
import networkx as nx
from tqdm import tqdm
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class MixedRates(nn.Module):
    #def __init__(self, n_items, init_rho1=0.8, init_rho2=0.2):
    def __init__(self, n_items, target_rho1=0.9, target_rho2=0.05):
        super(MixedRates, self).__init__()
        init_rho1 = math.log(target_rho1 / (1 - target_rho1))
        init_rho2 = math.log(target_rho2 / (1 - target_rho2))
        rates_init = torch.stack([
            torch.full((n_items,), init_rho1),
            torch.full((n_items,), init_rho2)
        ], dim=1)
        self.rates = nn.Parameter(rates_init)

    def forward(self):
        return torch.sigmoid(self.rates)

class Recommender(nn.Module):
    def __init__(self, n_user, n_item, n_entities, n_nodes, n_relations, args, graph, kg_sp, adj_mat, device):
        super(Recommender, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_entities = n_entities
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.emb_size = args.latdim

        self.node_dropout = args.node_dropout
        self.node_dropout_rate = args.node_dropout_rate
        self.mess_dropout = args.mess_dropout
        self.mess_dropout_rate = args.mess_dropout_rate

        self.context_hops = args.context_hops
        self.graph = graph
        self.device = device
        self.kg_sp = kg_sp
        self.inter_edge, self.inter_edge_w = self._convert_sp_mat_to_tensor(
            adj_mat)
        self.edge_index, self.edge_type = self._get_edges(graph)
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

        self.hgcn = AttnHGCN(channel=self.emb_size,
                             n_hops=self.context_hops,
                             n_users=self.n_user,
                             n_relations=self.n_relations,
                             n_item=self.n_item,
                             device=device,
                             node_dropout_rate=self.node_dropout_rate,
                             mess_dropout_rate=self.mess_dropout_rate)

        self.mixed_rates = MixedRates(self.n_item)

    def set_score_matrix(self, score_matrix):
        self.score_matrix = score_matrix

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))

    def get_rate(self):
        return self.mixed_rates().detach().cpu().numpy()

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def _convert_sp_mat_to_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return i.to(self.device), v.to(self.device)

    def _sparse_dropout(self, i, v, keep_rate=0.5):
        noise_shape = i.shape[1]

        random_tensor = keep_rate
        # the drop rate is 1 - keep_rate
        random_tensor += torch.rand(noise_shape).to(i.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)

        i = i[:, dropout_mask]
        v = v[dropout_mask] / keep_rate

        return i, v

    def _edge_sampling(self, edge_index, edge_type, samp_rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(
            n_edges, size=int(n_edges * samp_rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _relation_aware_edge_sampling(self, edge_index, edge_type, n_relations, samp_rate=0.5):
        # exclude interaction
        for i in range(n_relations - 1):
            edge_index_i, edge_type_i = self._edge_sampling(
                edge_index[:, edge_type == i], edge_type[edge_type == i], samp_rate)
            if i == 0:
                edge_index_sampled = edge_index_i
                edge_type_sampled = edge_type_i
            else:
                edge_index_sampled = torch.cat(
                    [edge_index_sampled, edge_index_i], dim=1)
                edge_type_sampled = torch.cat(
                    [edge_type_sampled, edge_type_i], dim=0)
        return edge_index_sampled, edge_type_sampled

    def pretrain_hgcn(self,adj_mat,n_item, args):

        #optimizer = torch.optim.Adam(self.hgcn.parameters(), lr=lr)
        lr = args.pretrain_lr
        pretrain_epochs = args.pretrain_epochs
        optimizer = torch.optim.Adam(list(self.hgcn.parameters()) + [self.all_embed], lr=lr)
        full_dataset = RecDataset(adj_mat)
        full_len = len(full_dataset)
        sample_size = args.sample_size

        self.train()
        for epoch in range(pretrain_epochs):
            if epoch % 5 == 0:
                sample_indices = np.random.choice(full_len, size=sample_size, replace=False)
                rec_dataset = RecDataset(adj_mat, sample_indices=sample_indices)

                rec_dataset.neg_sampling(n_item)
                rec_dataloader = DataLoader(rec_dataset, batch_size=args.rec_batch_size, shuffle=True)

            total_loss = 0.0
            for batch_idx, batch_data in tqdm(enumerate(rec_dataloader), total=len(rec_dataloader)):
                users, pos, neg = batch_data
                users, pos, neg = users.to(self.device), pos.to(self.device), neg.to(self.device)


                user_emb = self.all_embed[:self.n_user, :]
                entity_emb = self.all_embed[self.n_user:, :]

                inter_edge, inter_edge_w = self._sparse_dropout(
                    self.inter_edge, self.inter_edge_w, self.node_dropout_rate)

                edge_index, edge_type = self._relation_aware_edge_sampling(
                    self.edge_index, self.edge_type, self.n_relations, self.node_dropout_rate)


                entity_emb, user_emb = self.hgcn.shared_layer_agg(user_emb, entity_emb,edge_index,edge_type,inter_edge,inter_edge_w,self.device)

                item_emb = entity_emb[:self.n_item]

                batch_user_emb = user_emb[users.long()]
                pos_emb = item_emb[pos.long()]
                neg_emb = item_emb[neg.long()]

                pos_scores = torch.mul(batch_user_emb, pos_emb)
                pos_scores = torch.sum(pos_scores, dim=1)
                neg_scores = torch.mul(batch_user_emb, neg_emb)
                neg_scores = torch.sum(neg_scores, dim=-1)

                loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Pretrain HGCN Epoch {epoch + 1}/{pretrain_epochs}, Loss: {total_loss:.4f}")

    def get_entitiy_embeddings(self):
        return self.all_embed[self.n_user:, :].detach()

    def forward(self, batch):
        users, pos, neg = batch
        # users.to(self.device)
        # pos.to(self.device)
        # neg.to(self.device)
        users, pos, neg = users.to(self.device), pos.to(self.device), neg.to(self.device)

        user_emb = self.all_embed[:self.n_user, :]
        entity_emb = self.all_embed[self.n_user:, :]

        inter_edge, inter_edge_w = self._sparse_dropout(
            self.inter_edge, self.inter_edge_w, self.node_dropout_rate)

        rates = self.mixed_rates()
        dense_adj = torch.FloatTensor(self.kg_sp.toarray()).t()
        batch_items = pos.unique().long()
        kg_sp_csc = self.kg_sp.tocsc()
        batch_submatrix = kg_sp_csc[:, batch_items.cpu().numpy()]
        dense_adj_batch = torch.FloatTensor(batch_submatrix.toarray().T)
        rates_batch = rates[batch_items, :]
        total_predictions_batch = self.score_matrix[:, batch_items]
        adj_batch = knowledge_forward(total_predictions_batch, dense_adj_batch, rates_batch, self.device)
        edge_index, edge_type = self._relation_aware_edge_sampling(
            self.edge_index, self.edge_type, self.n_relations, self.node_dropout_rate)
        entity_emb, user_emb = self.hgcn(user_emb,
                                                   entity_emb,
                                                   edge_index,
                                                   edge_type,
                                                   inter_edge,
                                                   inter_edge_w,
                                                    adj_batch,
                                                    batch_items,
                                                   mess_dropout=self.mess_dropout,
                                                   )
        # print(entity_emb.device)
        # print(user_emb.device)
        item_emb = entity_emb[:self.n_item]

        batch_user_emb = user_emb[users.long()]
        pos_emb = item_emb[pos.long()]
        neg_emb = item_emb[neg.long()]

        pos_scores = torch.mul(batch_user_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(batch_user_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=-1)

        loss = torch.mean(nn.functional.softplus(neg_scores-pos_scores))
        #print("done forward!")
        return loss


def train_recommender(model, n_user, n_item, n_entities, n_nodes, n_relations, args, graph, kg_sp, adj_mat, device,
                      score_matrix):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    full_dataset = RecDataset(adj_mat)
    full_dataset.neg_sampling(n_item)
    rec_dataloader = DataLoader(full_dataset, batch_size=args.rec_batch_size, shuffle=True)

    model.to(device)
    model.train()

    for epoch in range(args.rec_epochs):
        total_loss = 0.0
        for batch_idx, batch_data in tqdm(enumerate(rec_dataloader), total=len(rec_dataloader)):
            loss = model(batch_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Training Recommender Epoch:{epoch + 1}, Loss:{total_loss}")

    rates = model.get_rate()
    return rates




