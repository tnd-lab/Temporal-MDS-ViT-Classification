import torch
import torch.nn as nn


class TemporalPatchEmbedding(nn.Module):
    def __init__(self, time_dim=55, img_size=256, in_channels=1, embed_dim=768):
        super().__init__()
        self.time_dim = time_dim
        self.img_size = img_size
        self.n_patches = (
            time_dim  # Number of patches = number of frames in the sequence
        )

        # Linear projection of flattened frames
        self.projection = nn.Linear(img_size * img_size * in_channels, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, time_dim, img_size, img_size)
        batch_size = x.shape[0]

        # Flatten each frame into a vector
        x = x.view(
            batch_size, self.time_dim, -1
        )  # Shape: (batch_size, time_dim, img_size * img_size * in_channels)

        # Linear projection to embedding dimension
        x = self.projection(x)  # Shape: (batch_size, time_dim, embed_dim)

        return x


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, n_tokens, embed_dim = x.shape

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(
                batch_size, n_tokens, self.num_heads, self.head_dim
            ).transpose(1, 2),
            qkv,
        )

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, n_tokens, embed_dim)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalMDSViT(nn.Module):
    def __init__(
        self,
        time_dim=55,
        img_size=256,
        in_channels=1,
        num_classes=10,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = TemporalPatchEmbedding(
            time_dim=time_dim,
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Add class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, time_dim + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        # Layer norm and classifier head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize patch embeddings and projection
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x shape: (batch_size, time_dim, img_size, img_size)
        batch_size = x.shape[0]

        # Create patch embeddings
        x = self.patch_embed(x)

        # Add class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add position embeddings
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Get class token output
        x = self.norm(x)
        x = x[:, 0]  # Take only the class token

        # Classification head
        x = self.head(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    time_dim = 55  # Number of frames in the sequence
    img_size = 256  # Height and width of each frame
    in_channels = 1  # Number of channels (grayscale images)
    num_classes = 3  # Number of classification classes
    embed_dim = 768  # Embedding dimension
    depth = 8  # Number of transformer layers
    num_heads = 12  # Number of attention heads
    mlp_ratio = 4.0  # MLP expansion ratio
    dropout = 0.1  # Dropout rate

    # Initialize the Temporal Vision Transformer
    model = TemporalMDSViT(
        time_dim=time_dim,
        img_size=img_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    )

    # Example Input Tensor
    # Shape: (batch_size, time_dim, img_size, img_size)
    batch_size = 4
    input_tensor = torch.randn(batch_size, time_dim, img_size, img_size)

    # Forward Pass
    output = model(input_tensor)

    # Output Shape
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)

    summary(model.to("cuda"), (55, 256, 256))
