import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.utils import softmax as scatter_softmax
import math
from logging import getLogger


class AttnHGCN(nn.Module):

    def __init__(self, channel, n_hops, n_users,
                 n_relations, n_item, device,
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(AttnHGCN, self).__init__()

        self.logger = getLogger()
        self.device = device
        self.n_item = n_item
        self.no_attn_convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        initializer = nn.init.xavier_uniform_
        relation_emb = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.relation_emb = nn.Parameter(relation_emb)  # [n_relations - 1, in_channel]

        self.W_Q = nn.Parameter(torch.Tensor(channel, channel))

        self.n_heads = 2
        self.d_k = channel // self.n_heads

        nn.init.xavier_uniform_(self.W_Q)

        self.n_hops = n_hops

        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def non_attn_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = relation_emb[
            edge_type - 1]
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """user aggregate"""
        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg

    def shared_layer_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, device):
        relation_emb = self.relation_emb
        n_entities = entity_emb.shape[0]

        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k).to(device)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k).to(device)

        key = key * relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k).to(device)

        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)

        relation_emb = relation_emb[edge_type - 1]
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k).to(device)

        # print(device)
        # print("-------------")
        # print(edge_attn_score.device)
        # print("============")
        # print(head.device)
        edge_attn_score = edge_attn_score.to(device)
        head = head.to(device)
        edge_attn_score = scatter_softmax(edge_attn_score, head)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads * self.d_k)
        # attn weight makes mean to sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)

        inter_edge = inter_edge.to(device)
        inter_edge_w = inter_edge_w.to(device)
        entity_emb = entity_emb.to(device)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        # w_attn = self.ui_weighting(user_emb, entity_emb, inter_edge)
        # item_agg += w_attn.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg

    def weighed_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb,
                    combined_adj, batch_item_ids):

        n_total = entity_emb.shape[0]
        head, tail = edge_index
        edge_relation_emb = relation_emb[edge_type - 1]
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb
        neigh_relation_emb.to(self.device)


        local_index = torch.full((n_total,), -1, dtype=torch.long, device=self.device)

        local_index[batch_item_ids] = torch.arange(batch_item_ids.size(0), device=self.device)

        edge_weight = torch.full(head.shape, 0.5, device=self.device, dtype=combined_adj.dtype)

        head = head.to(self.device)
        tail = tail.to(self.device)
        batch_item_ids = batch_item_ids.to(self.device)

        head_mask = torch.isin(head, batch_item_ids)
        tail_mask = torch.isin(tail, batch_item_ids)


        use_head_mask = head_mask

        use_tail_mask = (~head_mask) & tail_mask

        if use_head_mask.any():
            valid_head = head[use_head_mask]
            valid_tail = tail[use_head_mask]
            local_head_indices = local_index[valid_head]
            edge_weight[use_head_mask] = combined_adj[local_head_indices, valid_tail]

        if use_tail_mask.any():
            valid_tail = tail[use_tail_mask]
            valid_head = head[use_tail_mask]
            local_tail_indices = local_index[valid_tail]
            edge_weight[use_tail_mask] = combined_adj[local_tail_indices, valid_head]


        weighted_sum = scatter_sum(src=edge_weight.unsqueeze(-1) * neigh_relation_emb, index=head, dim_size=n_total,
                                   dim=0)
        weight_sum = scatter_sum(src=edge_weight, index=head, dim_size=n_total, dim=0)
        entity_agg = weighted_sum / (weight_sum.unsqueeze(-1) + 1e-9)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)

        return entity_agg, user_agg


    # @TimeCounter.count_time(warmup_interval=4)
    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                inter_edge, inter_edge_w, adj, batch_item_id, mess_dropout=True, item_attn=None):
        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        for i in range(self.n_hops):

            entity_emb, user_emb = self.weighed_agg(user_emb, entity_emb, edge_index, edge_type, inter_edge,inter_edge_w, self.relation_emb, adj, batch_item_id)
            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb

    def forward_ui(self, user_emb, item_emb, inter_edge, inter_edge_w, mess_dropout=True):
        item_res_emb = item_emb  # [n_entity, channel]
        for i in range(self.n_hops):
            user_emb, item_emb = self.ui_agg(user_emb, item_emb, inter_edge, inter_edge_w)
            """message dropout"""
            if mess_dropout:
                item_emb = self.dropout(item_emb)
                user_emb = self.dropout(user_emb)
            item_emb = F.normalize(item_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            item_res_emb = torch.add(item_res_emb, item_emb)
        return item_res_emb

    def forward_kg(self, entity_emb, edge_index, edge_type, mess_dropout=True):
        entity_res_emb = entity_emb
        for i in range(self.n_hops):
            entity_emb = self.kg_agg(entity_emb, edge_index, edge_type)
            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
        return entity_res_emb

    def ui_agg(self, user_emb, item_emb, inter_edge, inter_edge_w):
        num_items = item_emb.shape[0]
        item_emb = inter_edge_w.unsqueeze(-1) * item_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_emb, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        user_emb = inter_edge_w.unsqueeze(-1) * user_emb[inter_edge[0, :]]
        item_agg = scatter_sum(src=user_emb, index=inter_edge[1, :], dim_size=num_items, dim=0)
        return user_agg, item_agg

    def kg_agg(self, entity_emb, edge_index, edge_type):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        edge_relation_emb = self.relation_emb[
            edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        return entity_agg

    @torch.no_grad()
    def norm_attn_computer(self, entity_emb, edge_index, edge_type=None, print=False, return_logits=False):
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        if edge_type is not None:
            key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_logits = edge_attn.mean(-1).detach()
        # softmax by head_node
        edge_attn_score = scatter_softmax(edge_attn_logits, head)
        # normalization by head_node degree
        norm = scatter_sum(torch.ones_like(head), head, dim=0, dim_size=entity_emb.shape[0])
        norm = torch.index_select(norm, 0, head)
        edge_attn_score = edge_attn_score * norm
        # print attn score
        if print:
            self.logger.info("edge_attn_score std: {}".format(edge_attn_score.std()))
        if return_logits:
            return edge_attn_score, edge_attn_logits
        return edge_attn_score