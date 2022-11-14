import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


class PatchEncoder(nn.Module):
  def __init__(self, num_patches, channel_dim, proj_dim, device):
    super(PatchEncoder, self).__init__()
    self.num_patches = num_patches
    self.device = device
    self.projection = nn.Linear(channel_dim, proj_dim)
    self.pos_emb = nn.Embedding(num_patches, proj_dim)

  def forward(self, x):
    positions = torch.arange(0, self.num_patches, step=1, dtype=torch.int32, device=self.device)
    projection = self.projection(x) + self.pos_emb(positions)
    return projection


class Attention(nn.Module):
  def __init__(self, proj_dim, num_heads):
    super(Attention, self).__init__()
    self.layernorm = nn.LayerNorm(proj_dim)
    self.multihead = nn.MultiheadAttention(proj_dim, num_heads, dropout=0.1)
  
  def forward(self, x):
    x = self.layernorm(x)
    x, _ = self.multihead(x, x, x)
    return x


class MLP(nn.Module):
  def __init__(self, proj_dim, hidden_dim):
    super(MLP, self).__init__()
    self.layernorm = nn.LayerNorm(proj_dim)
    self.linear_1 = nn.Linear(proj_dim, hidden_dim)
    self.dropout_1 = nn.Dropout(0.1)
    self.linear_2 = nn.Linear(hidden_dim, proj_dim)
    self.dropout_2 = nn.Dropout(0.1)

  def forward(self, x):
    x = self.layernorm(x)
    x = self.linear_1(x)
    x = self.dropout_1(x)
    x = self.linear_2(x)
    x = self.dropout_2(x)
    return x


class Vit(nn.Module):
  def __init__(self,
                backbone_model,
                backbone_out_dims,
                num_layers,
                proj_dim,
                hidden_dim,
                num_heads,
                device,
                return_nodes):
    super(Vit, self).__init__()

    self.backbone_out_dims = backbone_out_dims
    self.feature_extractor = create_feature_extractor(backbone_model, return_nodes, suppress_diff_warning=True)
    self.patch_encoder = PatchEncoder(
        self.backbone_out_dims[-1],
        self.backbone_out_dims[0] * self.backbone_out_dims[1],
        proj_dim,
        device)
    self.transformer_layers = nn.ModuleList([
        nn.ModuleList([
            Attention(proj_dim, num_heads),
            MLP(proj_dim, hidden_dim)
        ])
    ]*num_layers)
    self.layernorm = nn.LayerNorm(proj_dim)
    self.linear = nn.Linear(proj_dim, 10)

  def forward(self, x):
    feature = self.feature_extractor(x)["fe_output"]
    b, c, h, w = feature.size()
    feature = feature.view(b, w, h*c)
    
    x = self.patch_encoder(feature)

    for attn, mlp in self.transformer_layers:
      x = x + attn(x)
      x = x + mlp(x)

    x = self.layernorm(x)

    x = x.mean(dim=1)
    
    x = self.linear(x)

    return x

