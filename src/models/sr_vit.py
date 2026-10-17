import torch
import torch.nn as nn
from src.models.sr import StackedSRNet

class PatchEmbedding(nn.Module):
    """
    Splits image into 2D patches and projects them to embed_dim.
    For a 32x96 SR image and 4x8 patch size: (32/4)*(96/8) = 8*12 = 96 patches.
    """
    def __init__(self, in_channels=3, patch_h=4, patch_w=8, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x) # [B, embed_dim, H/patch_h, W/patch_w]
        x = x.flatten(2).transpose(1, 2) # [B, N_patches, embed_dim]
        return x

class MultiFrameCrossAttentionFusion(nn.Module):
    """
    Spatio-Temporal Cross-Frame Attention.
    Center frame acts as Query. All frames act as Key/Value to borrow features.
    """
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, features, num_frames=5):
        # features: [B*F, N, embed_dim]
        B_F, N, C = features.size()
        B = B_F // num_frames
        
        # Reshape to [B, F, N, C]
        features = features.reshape(B, num_frames, N, C)
        
        # Center frame: Query -> [B, N, C]
        center_idx = num_frames // 2
        Q = features[:, center_idx, :, :] 
        
        # All frames concatenated spatially/temporally: Key/Value -> [B, F*N, C]
        KV = features.reshape(B, num_frames * N, C)
        
        # Cross Attention: Querying across space AND time
        attn_out, _ = self.cross_attn(Q, KV, KV)
        
        # Add & Norm
        out = self.norm(Q + attn_out) # [B, N, C]
        return out

class MultiFrameSRViT(nn.Module):
    """
    Flow 3: Pure Vision Transformer with Multi-Frame SR, Spatio-Temporal Cross-Attention,
    and Spatial-to-Sequence Height Pooling for 1D CTC alignment.
    """
    def __init__(
        self,
        num_classes: int,
        sr_num_blocks: int = 8,
        patch_h: int = 4,
        patch_w: int = 8,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 128 # Default max sequence length (e.g., handles up to 128 patches)
    ):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.embed_dim = embed_dim
        
        # 1. Frame-Parallel SR
        self.sr_net = StackedSRNet(num_blocks=sr_num_blocks, scale_factor=2)
        
        # 2. Patch Embedding
        self.patch_embed = PatchEmbedding(in_channels=3, patch_h=patch_h, patch_w=patch_w, embed_dim=embed_dim)
        
        # 3. Positional Encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # 4. Spatio-Temporal Cross-Attention Fusion
        self.fusion = MultiFrameCrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads)
        
        # 5. ViT Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        
        # 6. CTC Head
        self.head = nn.Linear(embed_dim, num_classes)
        # Backward compatibility alias
        self.ctc_head = self.head
        
    def forward(self, x, return_sr=False):
        # x: [B, F, 3, H, W]
        B, F, C, H, W = x.size()
        
        # 1. Super Resolution: [B, F, 3, H*2, W*2]
        sr_image = self.sr_net(x)
        
        # Flatten frames for Patch Embedding: [B*F, 3, H*2, W*2]
        _, _, C_sr, H_sr, W_sr = sr_image.size()
        sr_flat = sr_image.reshape(B * F, C_sr, H_sr, W_sr)
        
        # Calculate dynamic grid dimensions
        grid_h = H_sr // self.patch_h
        grid_w = W_sr // self.patch_w
        
        # 2. Patch Embedding: [B*F, N, embed_dim]
        patches = self.patch_embed(sr_flat)
        N = patches.size(1)
        
        # 3. Add Positional Encoding
        patches = patches + self.pos_embed[:, :N, :]
        
        # 4. Fusion: [B, N, embed_dim]
        fused_features = self.fusion(patches, num_frames=F)
        
        # 5. Transformer Encoder
        vit_out = self.encoder(fused_features)
        vit_out = self.norm(vit_out)
        
        # 6. Spatial-to-Sequence Height Pooling (Gộp chiều cao đưa về chuỗi 1D từ trái sang phải)
        # Reshape từ [B, Grid_H * Grid_W, D] về lưới 2D không gian: [B, Grid_H, Grid_W, D]
        grid_tokens = vit_out.view(B, grid_h, grid_w, self.embed_dim)
        # Average Pool dọc theo Grid_H -> [B, Grid_W, embed_dim]
        seq_tokens = torch.mean(grid_tokens, dim=1)
        
        # 7. CTC Head nhận chuỗi grid_w bước thời gian
        out = self.head(seq_tokens)
        log_probs = out.log_softmax(2)
        
        if return_sr:
            return log_probs, sr_image
        return log_probs
