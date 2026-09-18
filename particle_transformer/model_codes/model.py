import torch
import torch.nn as nn

from particle_embedding import ParticleEmbedding


class MLPHead(nn.Module):
    """
    MLP head that turns the CLS representation into the final classification logits
    """

    def __init__(self, embed_dim, hidden_dim=None, num_classes=2, dropout=0.1):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, embed_dim)
        return: (batch, num_classes)
        """
        return self.head(x)


class ParticleTransformerClassifier(nn.Module):
    """
    Overall model structure:
    batch_particle_dict
        -> ParticleEmbedding
        -> TransformerEncoder
        -> take the CLS token
        -> MLP classifier
        -> logits
    """

    def __init__(
        self,
        particle_dict,
        embed_dim=128,
        num_heads=8,
        num_layers=2,
        ff_dim=512,
        dropout=0.1,
        num_classes=2,
        head_hidden_dim=None,
        use_final_norm=True,
        use_positional_encoding=False,
    ):
        super().__init__()

        self.embedding = ParticleEmbedding(
            particle_dict=particle_dict,
            embed_dim=embed_dim,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.use_final_norm = use_final_norm
        if self.use_final_norm:
            self.final_norm = nn.LayerNorm(embed_dim)

        self.classifier = MLPHead(
            embed_dim=embed_dim,
            hidden_dim=head_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, particle_dict):
        """
        particle_dict[name]: (batch, feature_dim)

        return:
            logits: (batch, num_classes)
        """

        x = self.embedding(particle_dict)

        x = self.transformer(x)

        # take the CLS token: (batch, embed_dim)
        cls_repr = x[:, 0, :]

        if self.use_final_norm:
            cls_repr = self.final_norm(cls_repr)

        logits = self.classifier(cls_repr)

        return logits



if __name__ == "__main__":
    example_particle_dict = {
        "Xicplus": torch.randn(4, 12),
        "Ximinus": torch.randn(4, 10),
        "Lambda": torch.randn(4, 8),
        "Proton": torch.randn(4, 6),
        "Piminus": torch.randn(4, 5),
        "Pion0plus": torch.randn(4, 7),
        "Pion1plus": torch.randn(4, 7),
        "Xicplus_composite": torch.randn(4, 9),
    }

    model = ParticleTransformerClassifier(
        particle_dict=example_particle_dict,
        embed_dim=128,
        num_heads=8,
        num_layers=2,
        ff_dim=512,
        dropout=0.1,
        num_classes=2,
        head_hidden_dim=128,
        use_final_norm=True,
    )
    batch_particle_dict = {
        "Xicplus": torch.randn(32, 12),
        "Ximinus": torch.randn(32, 10),
        "Lambda": torch.randn(32, 8),
        "Proton": torch.randn(32, 6),
        "Piminus": torch.randn(32, 5),
        "Pion0plus": torch.randn(32, 7),
        "Pion1plus": torch.randn(32, 7),
        "Xicplus_composite": torch.randn(32, 9),
    }

    logits = model(batch_particle_dict)

    print("logits shape:", logits.shape)      # (32, 2)
