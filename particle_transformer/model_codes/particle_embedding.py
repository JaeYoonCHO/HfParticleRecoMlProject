import math
import torch
import torch.nn as nn


class ParticleEmbedding(nn.Module):

    def __init__(self, particle_dict, embed_dim=128, dropout=0.1, use_positional_encoding=False):
        super().__init__()

        # Mapping order defines token positions.
        self.particle_names = list(particle_dict)
        self.use_positional_encoding = use_positional_encoding
        self.num_particles = len(self.particle_names)
        self.embed_dim = embed_dim

        self.embedders = nn.ModuleDict()

        for name, tensor in particle_dict.items():
            input_dim = tensor.shape[1]
            self.embedders[name] = nn.Linear(input_dim, embed_dim)

        # CLS token: (1, 1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        pe = self._build_sincos_positional_encoding(
            seq_len=self.num_particles + 1,
            embed_dim=embed_dim
        )
        # Keep the buffer even when disabled for existing checkpoint compatibility.
        self.register_buffer("pos_encoding", pe)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        """
        Initialise the CLS token
        """
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _build_sincos_positional_encoding(self, seq_len, embed_dim):
        """
        Build the sinusoidal positional encoding
        return shape: (1, seq_len, embed_dim)
        """
        pe = torch.zeros(seq_len, embed_dim)

        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, embed_dim)
        return pe

    def forward(self, particle_dict):
        """
        particle_dict[name]: (batch, feature_dim)

        return:
            embedded_tensor: (batch, num_particles + 1, embed_dim)
            the first token is the CLS token
        """

        embeddings = []

        for name in self.particle_names:
            x = particle_dict[name]                # (batch, feature_dim)
            emb = self.embedders[name](x)         # (batch, embed_dim)
            embeddings.append(emb)

        embedded_tensor = torch.stack(embeddings, dim=1)

        batch_size = embedded_tensor.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        embedded_tensor = torch.cat([cls_tokens, embedded_tensor], dim=1)

        if self.use_positional_encoding:
            embedded_tensor = embedded_tensor + self.pos_encoding

        embedded_tensor = self.layer_norm(embedded_tensor)
        embedded_tensor = self.dropout(embedded_tensor)

        return embedded_tensor
