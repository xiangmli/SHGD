import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dataset import KGTripleDataset
from  torch.utils.data import DataLoader

class RelationPredictor(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0, norm=1):

        super(RelationPredictor, self).__init__()
        self.margin = margin
        self.norm = norm
        self.embedding_dim = embedding_dim


        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)


        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, heads, tails):


        h_embed = self.entity_embeddings(heads)  # (batch_size, embedding_dim)
        t_embed = self.entity_embeddings(tails)  # (batch_size, embedding_dim)


        h_expand = h_embed.unsqueeze(1)  # (batch_size, 1, embedding_dim)

        r_expand = self.relation_embeddings.weight.unsqueeze(0)
        t_expand = t_embed.unsqueeze(1)  # (batch_size, 1, embedding_dim)


        diff = h_expand + r_expand - t_expand  # (batch_size, num_relations, embedding_dim)

        distance = torch.norm(diff, p=self.norm, dim=2)  # (batch_size, num_relations)

        logits = -distance
        return logits




def train_relation_predictor(n_entities, n_relations, ckg_graph, args, device):

    model = RelationPredictor(n_entities, n_relations, args.embed_dim)
    dataset = KGTripleDataset(ckg_graph)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    num_epochs = args.rp_epochs
    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in dataloader:

            heads = batch['head'].to(device)
            tails = batch['tail'].to(device)
            labels = batch['relation'].to(device)

            optimizer.zero_grad()

            logits = model(heads, tails)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Training Relation Predictor Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model

def predict_relation(model, head, tail):
    model.eval()
    with torch.no_grad():
            logits = model(head, tail)
            predicted_relation = torch.argmax(logits, dim=1)
    return predicted_relation