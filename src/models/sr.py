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

class StackedSRNet(nn.Module):
    """
    Super-Resolution Network that stacks multiple LR frames to produce a single HR frame.
    Input: [Batch, Frames, 3, H, W] -> Stacked -> [Batch, 15, H, W]
    Output: [Batch, 3, H*scale, W*scale]
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
        self.in_channels = in_channels * num_frames
        self.scale_factor = scale_factor
        self.base_channels = base_channels
        
        # Initial feature extraction from stacked frames
        self.conv_first = nn.Conv2d(self.in_channels, base_channels, 3, padding=1, bias=True)
        
        # Residual body with RCAB blocks (EDSR style without BN + CALayer)
        self.body = nn.Sequential(*[
            RCAB(base_channels, base_channels, reduction=16) for _ in range(num_blocks)
        ])
        
        self.conv_body = nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=True)
        
        # Upsampling (PixelShuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * (scale_factor ** 2), 3, padding=1, bias=True),
            nn.PixelShuffle(scale_factor),
            nn.ReLU(inplace=True)
        )
        
        # Final reconstruction layer to get 3 channels output (RGB)
        self.conv_last = nn.Conv2d(base_channels, 3, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, N, C, H, W] where N = num_frames.
        Returns:
            HR image tensor of shape [B, 3, H*scale, W*scale].
        """
        B, N, C, H, W = x.size()
        
        # Stack frames along the channel dimension
        x = x.view(B, N * C, H, W)
        
        # Feature extraction
        x_first = self.conv_first(x)
        
        # Residual Body
        res = self.body(x_first)
        res = self.conv_body(res)
        res += x_first
        
        # Upsampling
        x_up = self.upsample(res)
        out = self.conv_last(x_up)
        
        return out
