import random
import torch
import numpy as np
import networkx as nx
import os
import numpy as np
import random
from tqdm import tqdm
from models.relation_predictor import predict_relation
import time
from models.AttnHGCN import AttnHGCN
from utils.dataset import RecDataset
from torch.utils.data import DataLoader

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_relation_for_pair(graph, u, v):
    data = graph.get_edge_data(u, v)
    if data:
        return list(data.keys())[0]
    return None

def precompute_item_neighbors(graph, n_item, num_nodes):
    item_neighbors = {}
    for i in range(n_item):
        neighbors = {i}
        for node in graph.successors(i):
            if node < num_nodes:
                neighbors.add(node)
        item_neighbors[i] = list(neighbors)
    return item_neighbors

def denoise_graph(total_predictions, graph, n_item, rho1=0.8, keep_entities=False):

    denoised_graph = nx.MultiDiGraph()

    #for i in range(n_item):
    for i in tqdm(range(n_item), desc="Denoising graph for items", unit="item"):

        original_entities = list(graph.successors(i))
        if not original_entities:
            continue

        k = len(original_entities)
        num_to_keep = int(rho1 * k)

        if num_to_keep < 1:
            num_to_keep = 1


        candidate_scores = [(total_predictions[entity, i], entity) for entity in original_entities]

        candidate_scores.sort(key=lambda x: x[0], reverse=True)

        kept_candidates = candidate_scores[:num_to_keep]

        for score, entity in kept_candidates:
            r = get_relation_for_pair(graph, i, entity)
            if r is not None:
                #print(r)
                denoised_graph.add_edge(i, entity, key=r)
            else:
                print("r is None!!!!")

    if keep_entities:
        for u, v, r in graph.edges(data='key'):
            if u >= n_item and v >= n_item:
                denoised_graph.add_edge(u, v, key=r)

    return denoised_graph


def complete_graph(total_predictions, graph, n_item, relation_predictor, device, rho2=0.2):

    #completed_graph = nx.MultiDiGraph()
    completed_graph = graph.copy()


    for i in tqdm(range(n_item), desc="Completing graph for items", unit="item"):

        original_entities = set(graph.successors(i))
        if not original_entities:

            continue


        all_entities = np.arange(n_item, total_predictions.shape[0])

        candidate_mask = np.isin(all_entities, list(original_entities), invert=True)
        candidate_entities = all_entities[candidate_mask]

        if candidate_entities.size == 0:
            continue


        num_to_add = int(rho2 * len(original_entities))
        if num_to_add < 1:
            num_to_add = 1

        num_to_add = min(num_to_add, candidate_entities.size)

        candidate_scores = total_predictions[candidate_entities, i]


        top_indices = np.argpartition(-candidate_scores, num_to_add - 1)[:num_to_add]

        sorted_top = top_indices[np.argsort(-candidate_scores[top_indices])]
        selected_entities = candidate_entities[sorted_top]


        selected_entities = np.atleast_1d(selected_entities)


        for entity in selected_entities:
            head = torch.tensor([i], dtype=torch.long).to(device)
            tail = torch.tensor([entity], dtype=torch.long).to(device)


            predicted_relation = relation_predictor.predict_relation(head, tail)

            completed_graph.add_edge(i, entity, key=predicted_relation.item())

    return completed_graph

def denoise_graph_adapt(total_predictions, graph, n_item, rho1_vector, keep_entities=False):

    denoised_graph = nx.MultiDiGraph()
    print(rho1_vector)

    #for i in range(n_item):
    for i in tqdm(range(n_item), desc="Denoising graph for items", unit="item"):

        if graph.has_node(i):
            original_entities = list(graph.successors(i))
        #original_entities = list(graph.successors(i))
        if not original_entities:
            continue

        k = len(original_entities)
        #num_to_keep = int(rho1 * k)
        num_to_keep = int(rho1_vector[i] * k)

        if num_to_keep < 1:
            num_to_keep = 1


        candidate_scores = [(total_predictions[entity, i], entity) for entity in original_entities]

        candidate_scores.sort(key=lambda x: x[0], reverse=True)

        kept_candidates = candidate_scores[:num_to_keep]

        for score, entity in kept_candidates:
            r = get_relation_for_pair(graph, i, entity)
            if r is not None:
                #print(r)
                denoised_graph.add_edge(i, entity, key=r)
            else:
                print("r is None!!!!")

    if keep_entities:
        for u, v, r in graph.edges(data='key'):
            if u >= n_item and v >= n_item:
                denoised_graph.add_edge(u, v, key=r)

    return denoised_graph

def complete_graph_adapt(total_predictions, graph, n_item, relation_predictor, device, rho2_vector):

    #completed_graph = nx.MultiDiGraph()
    completed_graph = nx.MultiDiGraph()
    print(rho2_vector)

    for i in tqdm(range(n_item), desc="Completing graph for items", unit="item"):

        if graph.has_node(i):
            original_entities = list(graph.successors(i))
        #original_entities = set(graph.successors(i))
        if not original_entities:

            continue


        #all_entities = np.arange(total_predictions.shape[0])
        all_entities = np.arange(n_item, total_predictions.shape[0])

        candidate_mask = np.isin(all_entities, list(original_entities), invert=True)
        candidate_entities = all_entities[candidate_mask]

        if candidate_entities.size == 0:
            continue


        num_to_add = int(rho2_vector[i] * len(original_entities))
        if num_to_add < 1:
            num_to_add = 1
        num_to_add = min(num_to_add, candidate_entities.size)


        candidate_scores = total_predictions[candidate_entities, i]


        top_indices = np.argpartition(-candidate_scores, num_to_add - 1)[:num_to_add]

        sorted_top = top_indices[np.argsort(-candidate_scores[top_indices])]
        selected_entities = candidate_entities[sorted_top]

        selected_entities = np.atleast_1d(selected_entities)

        # for entity in selected_entities:
        #     completed_graph.add_edge(i, entity, key=0)
        for entity in selected_entities:
            head = torch.tensor([i], dtype=torch.long).to(device)
            tail = torch.tensor([entity], dtype=torch.long).to(device)


            predicted_relation = predict_relation(relation_predictor, head, tail)
            #print(predicted_relation)

            completed_graph.add_edge(i, entity, key=predicted_relation.item())

    return completed_graph



def combine_graphs(denoised_graph, completed_graph):

    combined_graph = nx.MultiDiGraph()

    for u, v, k, data in tqdm(denoised_graph.edges(keys=True, data=True), desc="Processing denoised edges",
                              unit="edge"):
        combined_graph.add_edge(u, v, key=k, **data)


    for u, v, k, data in tqdm(completed_graph.edges(keys=True, data=True), desc="Processing completed edges",
                              unit="edge"):
        combined_graph.add_edge(u, v, key=k, **data)
    return combined_graph


def binary_concrete_sample(logits, temperature=0.5):

    eps = 1e-20
    noise = torch.rand_like(logits)
    logistic_noise = torch.log(noise + eps) - torch.log(1 - noise + eps)
    y = logits + logistic_noise
    return torch.sigmoid(y / temperature)




def differentiable_denoise(total_predictions, dense_adj, rho1_vector, device, temperature=0.5, alpha=1.0):
    alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
    scores = total_predictions.T
    scores = scores - scores.mean(dim=1, keepdim=True)
    #print(device)
    scores = scores.to(device)


    bias = torch.log(rho1_vector / (1 - rho1_vector)).unsqueeze(1)  # shape: (n_item, 1)
    # print(f"alpha device: {alpha.device}")
    # print(f"scores device: {scores.device}")
    # print(f"bias device: {bias.device}")

    logits = alpha * scores + bias  # shape: (n_item, n_entities)
    #print("done calculating logits")

    mask = binary_concrete_sample(logits, temperature=temperature)
    #print("done samping")

    dense_adj = dense_adj.to(device)
    new_adj = dense_adj * mask
    #print("done new adj1")
    return new_adj


def differentiable_complete(total_predictions, dense_adj, rho2_vector, device,temperature=0.5, alpha=1.0):
    alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
    scores = total_predictions.T
    scores = scores - scores.mean(dim=1, keepdim=True)
    scores = scores.to(device)
    dense_adj = dense_adj.to(device)
    candidate_mask = 1 - dense_adj


    masked_scores = scores * candidate_mask + (1 - candidate_mask) * (-1e9)


    bias = torch.log(rho2_vector / (1 - rho2_vector)).unsqueeze(1)  # shape: (n_item, 1)
    #bias = bias.to('cpu')

    logits = alpha * masked_scores + bias  # shape: (n_item, n_entities)
    #print("done calculating logits")
    # Binary Concrete 采样，得到新边生成的“概率”
    mask = binary_concrete_sample(logits, temperature=temperature)
    #print("done sampling")
    new_adj = candidate_mask * mask
    #print("done calculating adj2")
    return new_adj


def knowledge_forward(total_predictions, dense_adj, rates,device,
                                        temperature=0.5, alpha=1.0):


    rho1_vector = rates[:,0]
    rho2_vector = rates[:,1]
    denoised_adj = differentiable_denoise(total_predictions, dense_adj, rho1_vector, device,temperature, alpha)


    completed_adj = differentiable_complete(total_predictions, dense_adj, rho2_vector, device,temperature, alpha)

    combined_adj = denoised_adj + completed_adj
    #print("done combining!")

    return combined_adj

def save_knowledge_graph(kg, graph_name, dataset_name):

    save_path = f'./generated_kg/{dataset_name}/{graph_name}'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    with open(save_path, 'w') as f:

        #for h, t, key, data in kg.edges(keys=True, data=True):
        for h, t, key, data in tqdm(kg.edges(keys=True, data=True), desc="Saving knowledge graph", unit="edge"):

            f.write(f"{h} {key} {t}\n")

    print(f"knowledge graph saved at: {save_path}")
