
import random
import torch
import numpy as np
import os
from time import time
import utils.helpers
import datetime
from utils.parser import parse_args
from utils.data_loader import load_data
from models.giffcf import SHGD
from models.relation_predictor import train_relation_predictor
from models.recommender import train_recommender
from utils.dataset import KGDataset
from torch.utils.data import DataLoader
from utils.helpers import denoise_graph_adapt, complete_graph_adapt, combine_graphs, save_knowledge_graph
from models.recommender import Recommender
seed = 2025
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def main():
    seed = 2025
    utils.helpers.seed_everything(seed)

    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    print(f"device:{device}")
    train_cf, train_cf_sp, test_cf, user_dict, n_params, graph, kg_sp, adj_mat = load_data(args)
    train_dataset = KGDataset(kg_sp)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    infer_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    print("data ready!")



    epochs = args.epochs

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_nodes = n_params['n_nodes']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']

    recommender = Recommender(n_users, n_items, n_entities, n_nodes, n_relations, args, graph, kg_sp, train_cf_sp, device)
    recommender.to(device)
    recommender.pretrain_hgcn(train_cf_sp, n_items, args)
    entity_embeddings = recommender.get_entitiy_embeddings()

    model = SHGD(train_cf_sp.toarray(), kg_sp, entity_embeddings, device,args)

    model = model.to(device)
    print("model ready!")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            loss = model(batch_data, batch_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss

        print(f"Epoch[{epoch+1}/{epochs}], Loss:{total_loss}")

    print("Done Training")


    total_predictions = torch.zeros(n_entities, n_items)


    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(infer_loader):
            batch_data = batch_data.to(device)
            pred = model(batch_data, batch_idx,training=False)
            start_idx = batch_idx * args.batch_size
            end_idx = min((batch_idx + 1) * args.batch_size, n_entities)
            total_predictions[start_idx:end_idx] = pred.cpu()
    recommender.set_score_matrix(total_predictions)
    #if args.mode == "denoise":
    #new_kg = utils.helpers.denoise_graph(total_predictions, graph, n_items,rho1=args.denoise_keep_rate)
    #elif args.mode == "complete":
    relation_predictor = train_relation_predictor(n_entities, n_relations, graph, args, device)
    #new_kg2 = utils.helpers.complete_graph(total_predictions, graph, n_items, relation_predictor, device, rho2=args.complete_rate)

    # utils.helpers.save_knowledge_graph(new_kg, 'denoised_graph.txt', args.dataset)
    # utils.helpers.save_knowledge_graph(new_kg2, 'complete_graph.txt', args.dataset)

    rates = train_recommender(recommender, n_users, n_items, n_entities, n_nodes, n_relations, args, graph, kg_sp, train_cf_sp, device, total_predictions)

    denoised_graph = denoise_graph_adapt(total_predictions, graph, n_items, rates[:,0], keep_entities=True)
    completed_graph = complete_graph_adapt(total_predictions, graph, n_items, relation_predictor, device, rates[:, 1])
    knowledge_final = combine_graphs(denoised_graph, completed_graph)
    save_knowledge_graph(knowledge_final, 'generated_kg100_alpha=2.5.txt', args.dataset)

if __name__ == '__main__':
    main()