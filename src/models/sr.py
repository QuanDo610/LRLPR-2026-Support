import torch
import torch.nn as nn
import torch.nn.functional as F


class CALayer(nn.Module):
    """Channel Attention (CA) Layer from RCAN."""
    def __init__(self, channel: int, reduction: int = 16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) - EDSR style without BN + CALayer."""
    def __init__(self, in_feat: int, out_feat: int, reduction: int = 16):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_feat, out_feat, 3, padding=1, bias=True)
        self.ca = CALayer(out_feat, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.ca(res)
        res += x
        return res


class CrossFrameTemporalSRFusion(nn.Module):
    """
    Multi-Frame Spatio-Temporal Feature Alignment & Fusion for Super-Resolution.
    Exchanges complementary sub-pixel stroke information across the N frames of a track
    prior to PixelShuffle upsampling. Zero-initialized output projection preserves
    100% backward compatibility with existing checkpoints at Epoch 0.
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.spatial_align = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, groups=channels, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.attn_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, num_frames: int = 5) -> torch.Tensor:
        BN, C, H, W = x.size()
        if num_frames <= 1 or BN % num_frames != 0:
            return x
        B = BN // num_frames
        feat = x.view(B, num_frames, C, H, W)

        # Pairwise cross-frame alignment between target frame i and support frames j
        f_i = feat.unsqueeze(2).expand(B, num_frames, num_frames, C, H, W)
        f_j = feat.unsqueeze(1).expand(B, num_frames, num_frames, C, H, W)
        pairs = torch.cat([f_i, f_j], dim=3).reshape(B * num_frames * num_frames, C * 2, H, W)

        aligned = self.spatial_align(pairs).view(B, num_frames, num_frames, C, H, W)
        weights = self.attn_gate(pairs).view(B, num_frames, num_frames, C, H, W)
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-6)

        fused = (aligned * weights).sum(dim=2).reshape(BN, C, H, W)
        return x + self.out_proj(fused)


class StackedSRNet(nn.Module):
    """
    Multi-Frame Super-Resolution Network with Cross-Frame Spatio-Temporal Alignment.
    Input: [Batch, Frames, 3, H, W] -> Output: [Batch, Frames, 3, H*scale, W*scale]
    Or Input: [Batch*Frames, 3, H, W] -> Output: [Batch*Frames, 3, H*scale, W*scale]
    """
    def __init__(
        self,
        num_frames: int = 5,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 8,
        scale_factor: int = 2
    ):
        super(StackedSRNet, self).__init__()
        self.num_frames = num_frames
        self.scale_factor = scale_factor

        # Initial feature extraction
        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=True)

        # Residual body with RCAB blocks
        self.body = nn.Sequential(*[
            RCAB(base_channels, base_channels, reduction=16) for _ in range(num_blocks)
        ])

        self.conv_body = nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=True)

        # Cross-frame spatio-temporal alignment before upsampling
        self.temporal_sr_fusion = CrossFrameTemporalSRFusion(base_channels)

        # Upsampling (PixelShuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * (scale_factor ** 2), 3, padding=1, bias=True),
            nn.PixelShuffle(scale_factor),
            nn.ReLU(inplace=True)
        )

        # Final reconstruction layer
        self.tail = nn.Conv2d(base_channels, 3, 3, padding=1, bias=True)
        self.res_scale_gate = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_5d = x.dim() == 5
        n_frames = x.size(1) if is_5d else self.num_frames
        if is_5d:
            B, N, C, H, W = x.size()
            x = x.view(B * N, C, H, W)

        # Global residual connection via bicubic interpolation
        base = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

        feat = self.head(x)
        res = self.body(feat)
        res = self.conv_body(res)
        if is_5d:
            res = self.temporal_sr_fusion(res, num_frames=n_frames)
        res = res + feat

        out = self.tail(self.upsample(res))
        out = torch.clamp(base + self.res_scale_gate * out, -1.0, 1.0)

        if is_5d:
            _, C_out, H_out, W_out = out.size()
            out = out.view(B, N, C_out, H_out, W_out)

        return out
