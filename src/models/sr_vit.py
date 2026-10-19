import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import STNBlock
from src.models.sr import StackedSRNet


class PatchEmbedding(nn.Module):
    """
    Overlapping Convolutional Stem + 2D Patch Projection with LayerNorm.
    For a 32x128 SR image and 4x8 patch size: (32/4)*(128/8) = 8*16 = 128 patches.
    Preserves stroke continuity across patch boundaries while maintaining exact [B*F, N, embed_dim] shape.
    """
    def __init__(self, in_channels: int = 3, patch_h: int = 4, patch_w: int = 8, embed_dim: int = 256):
        super().__init__()
        half_dim = embed_dim // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, half_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(half_dim),
            nn.GELU(),
            nn.Conv2d(half_dim, half_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(half_dim),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(half_dim, embed_dim, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*F, C, H, W]
        x = self.stem(x)
        x = self.proj(x)  # [B*F, embed_dim, H/patch_h, W/patch_w]
        x = x.flatten(2).transpose(1, 2)  # [B*F, N_patches, embed_dim]
        return self.norm(x)


class MultiFrameCrossAttentionFusion(nn.Module):
    """
    Spatio-Temporal Cross-Frame Attention with learnable Temporal Frame Embeddings.
    Center frame acts as Query. All frames act as Key/Value to borrow sharp character strokes.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, max_frames: int = 5):
        super().__init__()
        self.frame_embed = nn.Parameter(torch.zeros(1, max_frames, 1, embed_dim))
        nn.init.trunc_normal_(self.frame_embed, std=0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features: torch.Tensor, num_frames: int = 5) -> torch.Tensor:
        # features: [B*F, N, embed_dim]
        B_F, N, C = features.size()
        B = B_F // num_frames

        # Reshape to [B, F, N, C] and inject temporal frame identity
        features = features.reshape(B, num_frames, N, C)
        if num_frames <= self.frame_embed.size(1):
            features = features + self.frame_embed[:, :num_frames, :, :]

        # Center frame: Query -> [B, N, C]
        center_idx = num_frames // 2
        Q = features[:, center_idx, :, :]

        # All frames concatenated spatially/temporally: Key/Value -> [B, F*N, C]
        KV = features.reshape(B, num_frames * N, C)

        # Cross Attention: Querying across space AND time
        attn_out, _ = self.cross_attn(Q, KV, KV)

        # Add & Norm
        out = self.norm(Q + attn_out)  # [B, N, C]
        return out


class MultiFrameSRViT(nn.Module):
    """
    Flow 3: Vision Transformer with Frame-Parallel SR, STN Alignment,
    Spatio-Temporal Cross-Attention, and Spatial-to-Sequence Height Pooling for 1D CTC alignment.
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
        max_seq_len: int = 128,
        use_stn: bool = True,
    ):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.embed_dim = embed_dim
        self.use_stn = use_stn

        # 1. Frame-Parallel SR
        self.sr_net = StackedSRNet(num_blocks=sr_num_blocks, scale_factor=2)

        # 2. Spatial Transformer Network (STN) on Frame-Parallel SR outputs (Review3.md Table 3)
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 3. Overlapping Patch Embedding
        self.patch_embed = PatchEmbedding(
            in_channels=3, patch_h=patch_h, patch_w=patch_w, embed_dim=embed_dim
        )

        # 4. 2D Positional Encoding (initialized with trunc_normal_ to break vertical symmetry)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 5. Spatio-Temporal Cross-Attention Fusion
        self.fusion = MultiFrameCrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads)

        # 6. ViT Encoder (with GELU activation)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # 7. Vertical Row Weighting before Height Pooling
        self.row_proj = nn.Linear(embed_dim, 1)
        nn.init.zeros_(self.row_proj.weight)
        nn.init.zeros_(self.row_proj.bias)

        # 8. CTC Head
        self.head = nn.Linear(embed_dim, num_classes)
        self.ctc_head = self.head

    def forward(self, x: torch.Tensor, return_sr: bool = False):
        # x: [B, F, 3, H, W]
        B, F_frames, C, H, W = x.size()

        # 1. Super Resolution: [B, F, 3, H*2, W*2]
        sr_conv_trainable = any(
            p.requires_grad for n, p in self.sr_net.named_parameters() if n != "res_scale_gate"
        )
        if sr_conv_trainable or self.sr_net.res_scale_gate.requires_grad:
            sr_image = self.sr_net(x)
        else:
            with torch.no_grad():
                sr_image = self.sr_net(x)

        _, _, C_sr, H_sr, W_sr = sr_image.size()
        sr_flat = sr_image.reshape(B * F_frames, C_sr, H_sr, W_sr)

        # 2. STN Spatial Alignment across all 5 SR frames
        if self.use_stn:
            theta = self.stn(sr_flat)
            grid = F.affine_grid(theta, sr_flat.size(), align_corners=False)
            sr_aligned = F.grid_sample(sr_flat, grid, align_corners=False)
        else:
            sr_aligned = sr_flat

        grid_h = H_sr // self.patch_h
        grid_w = W_sr // self.patch_w

        # 3. Patch Embedding: [B*F, N, embed_dim]
        patches = self.patch_embed(sr_aligned)
        N = patches.size(1)

        # 4. Add Positional Encoding (with dynamic interpolation fallback if N != max_seq_len)
        if N <= self.pos_embed.size(1):
            pos = self.pos_embed[:, :N, :]
        else:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2), size=N, mode='linear', align_corners=False
            ).transpose(1, 2)
        patches = patches + pos

        # 5. Spatio-Temporal Cross-Attention Fusion: [B, N, embed_dim]
        fused_features = self.fusion(patches, num_frames=F_frames)

        # 6. Transformer Encoder
        vit_out = self.encoder(fused_features)
        vit_out = self.norm(vit_out)

        # 7. Spatial-to-Sequence Height Pooling: [B, Grid_H, Grid_W, D] -> [B, Grid_W, D]
        grid_tokens = vit_out.view(B, grid_h, grid_w, self.embed_dim)
        row_attn = torch.softmax(self.row_proj(grid_tokens), dim=1)  # [B, Grid_H, Grid_W, 1]
        seq_tokens = torch.sum(grid_tokens * row_attn, dim=1)  # Uniform mean at init, adaptive row pooling after training

        # 8. CTC Head in FP32
        out = self.head(seq_tokens)
        log_probs = out.float().log_softmax(2)

        if return_sr:
            return log_probs, sr_image
        return log_probs
