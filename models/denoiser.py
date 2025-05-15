import torch
import torch.nn as nn

import numpy as np


class Timestep(nn.Module):
    def __init__(self, embed_dim, n_steps, max_wavelength=10000.0):
        assert embed_dim % 2 == 0
        super(Timestep, self).__init__()
        timescales = np.power(max_wavelength, -np.arange(0, embed_dim, 2) / embed_dim)
        timesteps = np.arange(n_steps + 1)
        angles = timesteps[:, np.newaxis] * timescales[np.newaxis, :]
        sinusoids = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
        self.register_buffer('sinusoids', torch.tensor(sinusoids, dtype=torch.float32))

    def forward(self, timesteps):
        return self.sinusoids[timesteps]


class TimeEmbed(nn.Module):
    def __init__(self, hidden_dim, out_dim, activation, n_steps):
        super(TimeEmbed, self).__init__()
        self.timestep = Timestep(hidden_dim, n_steps)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.activation = activation

    def forward(self, t):
        e = self.timestep(t)
        x = self.hidden(e)
        if self.activation == 'swish':
            x = x * torch.sigmoid(x)
        return self.out(x)


class SimpleMixer(nn.Module):
    def __init__(self, hidden_dim, activation):
        super(SimpleMixer, self).__init__()
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.activation = activation

    def forward(self, inputs):
        x = torch.stack(inputs, dim=-1)  #[batch_size, emb_size, 3]
        x = self.hidden(x)
        if self.activation == 'swish':
            x = x * torch.sigmoid(x)
        x = self.out(x)
        return torch.squeeze(x, dim=-1)

class ScoreMixer(nn.Module):
    def __init__(self, hidden_dim, n_items, embed_dim, activation):
        super(ScoreMixer, self).__init__()
        self.embed_dim = embed_dim
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.activation = activation
        self.t_embed2_proj = nn.Linear(self.embed_dim, n_items)  # n_items 即 c.shape[1]

    def forward(self, inputs):
        t_embed2_proj = self.t_embed2_proj(inputs[-1])
        x = torch.stack(inputs[:-1], dim=-1)  #[batch_size, emb_size, 3]
        t_embed2_proj = t_embed2_proj.unsqueeze(-1) #[batch_size, emb_size, 1]
        x = torch.cat([x, t_embed2_proj], dim=-1) #[batch_size, emb_size, 4]

        x = self.hidden(x)
        if self.activation == 'swish':
            x = x * torch.sigmoid(x)
        x = self.out(x)
        return torch.squeeze(x, dim=-1)

class Denoiser(nn.Module):
    def __init__(
        self,
        n_items,
        embed_dim=200,
        activation='swish',
        initializer='glorot_uniform',
        norm_ord=1,
        n_steps=10,
        ablation=None,
    ):
        super(Denoiser, self).__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.norm_ord = norm_ord
        self.item_embed = nn.Parameter(torch.randn(n_items, embed_dim))
        self.t_embed1 = TimeEmbed(20, 1, activation, n_steps)
        self.t_embed2 = TimeEmbed(20, 1, activation, n_steps)
        self.embed_mixer = SimpleMixer(3, activation)
        self.score_mixer = ScoreMixer(4, self.n_items, self.embed_dim, activation)
        self.ablation = ablation

    @property
    def mlp_weights(self):
        return (
            list(self.t_embed1.parameters())
            + list(self.t_embed2.parameters())
            + list(self.embed_mixer.parameters())
            + list(self.score_mixer.parameters())
        )

    def norm(self, c):
        if self.norm_ord is not None:
            norm = torch.norm(c, p=self.norm_ord, dim=-1, keepdim=True)
            return torch.maximum(norm, torch.tensor(1.0))
        else:
            return 1.0

    def forward(self, z_t, c, Ac, t, training=None):
        #t_embed1 = self.t_embed1(t).repeat(1, self.embed_dim)
        #t_embed2 = self.t_embed2(t).repeat(1, self.n_items)
        #print(self.t_embed1(t).shape)
        t_embed1 = self.t_embed1(t).squeeze(2)
        t_embed1 = t_embed1.repeat(1, self.embed_dim)
        t_embed2 = self.t_embed2(t).squeeze(2)
        t_embed2 = t_embed2.repeat(1, self.embed_dim)

        z_embed = torch.matmul(z_t / self.n_items, self.item_embed)
        c_embed = torch.matmul(c / self.norm(c), self.item_embed)
        if self.ablation == 'wo_latent':
            x_embed = self.embed_mixer([c_embed, t_embed1])
        elif self.ablation == 'wo_precond':
            x_embed = self.embed_mixer([z_embed, t_embed1])
        else:
            x_embed = self.embed_mixer([z_embed, c_embed, t_embed1])  # [batch_size, emb_dim]
        # print(x_embed.shape)  #[batch_size, emb_dim]
        # print(self.item_embed.shape)  # [n_item, emb_dim]
        x_mid = torch.matmul(x_embed, self.item_embed.t())
        if self.ablation == 'wo_postcond':
            x_pred = self.score_mixer([x_mid, t_embed2])
        else:
            # print(x_mid.shape)  #[batch_size. n_item]
            # print(c.shape)  #[batch_size, n_item]
            # print(Ac.shape)  # [batch_size, n_item]
            # print(t_embed2.shape) # [batch_size, emb_dim]
            x_pred = self.score_mixer([x_mid, c, Ac, t_embed2])
        return x_pred



class MLP_Denoiser(nn.Module):
    def __init__(self, n_item, hidden_layers, n_steps, activation='swish', time_embed_hidden_dim=20):

        super(MLP_Denoiser, self).__init__()

        self.time_embed = TimeEmbed(time_embed_hidden_dim, 1, activation, n_steps)


        mlp_input_dim = n_item + 1

        layers = []
        input_dim = mlp_input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        layers.append(nn.Linear(input_dim, n_item))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, c, Ac, t, training=None):


        if t.dim() == 1:
            t = t.unsqueeze(1)

        t_encoded = self.time_embed(t)

        t_encoded = t_encoded.squeeze(-1)

        x_cat = torch.cat([x, t_encoded], dim=1)

        return self.mlp(x_cat)