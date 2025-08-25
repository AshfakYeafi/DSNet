"""
DSNet 3D architecture
------------------------------------------------
Modules:
- attention3d: produces soft attention over K dynamic kernels (per-sample).
- Dynamic_conv3d: soft mixture of K learnable 3D convolution kernels.
- Attention_block: attention gate for UNet skip connections (3D).
- ResDoubleConv: residual double dynamic-conv block (3D).
- ResDown: encoder step (MaxPool3d + ResDoubleConv).
- up_conv: decoder upsampling step (ConvTranspose3d).
- Out: 1x1x1 dynamic conv projection to class logits.
- ResAttUnetDync: UNet-style encoder-decoder with attention skips and dynamic convs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable  # kept for full compatibility (not used directly)

import warnings
warnings.simplefilter("ignore")


# ---------------------------------------------------------------------
# Dynamic convolution: attention over K candidate 3D kernels (per sample)
# ---------------------------------------------------------------------
# @https://github.com/kaijieshi7/Dynamic-convolution-Pytorch
class attention3d(nn.Module):
    """
    Compute attention weights over K candidate kernels given input x.
    The attention is global (spatially pooled) and channel-aware.

    Args:
        in_planes (int):   input channels.
        ratios (float):    reduction ratio for hidden size in attention MLP.
        K (int):           number of dynamic kernels to mix.
        temperature (int): softmax temperature; original schedule expects (temp % 3 == 1).
    """
    def __init__(self, in_planes, ratios, K, temperature):
        super(attention3d, self).__init__()
        assert temperature % 3 == 1

        # Global pooling to summarize spatial context
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # Hidden size rule from original impl.
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K

        # Two 1x1x1 convs act like an MLP on pooled channels
        self.fc1 = nn.Conv3d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv3d(hidden_planes, K, 1, bias=False)
        self.temperature = temperature

    def updata_temperature(self):
        """
        Decrease temperature by 3 (down to 1) as per original implementation.
        NOTE: name kept as-is for compatibility.
        """
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        """
        x:  [B, C, D, H, W]
        return: attention weights over K kernels, shape [B, K], sums to 1 over K.
        """
        x = self.avgpool(x)              # [B, C, 1, 1, 1]
        x = self.fc1(x)                  # [B, hidden, 1, 1, 1]
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)  # [B, K]
        return F.softmax(x / self.temperature, 1)


class Dynamic_conv3d(nn.Module):
    """
    Dynamic 3D convolution (mixture-of-experts):
    For each sample, compute attention over K candidate 3D kernels, then
    form a single aggregated kernel and bias to apply via grouped conv.

    Args mirror nn.Conv3d where applicable, plus:
        ratio, K, temperature: control attention capacity/behavior.
    """
    def __init__(self, in_planes, out_planes, kernel_size,
                 ratio=0.25, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, K=4, temperature=34):
        super(Dynamic_conv3d, self).__init__()
        assert in_planes % groups == 0

        # Save shape/conv settings
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K

        # Attention that produces [B, K] weights
        self.attention = attention3d(in_planes, ratio, K, temperature)

        # K candidate kernels: [K, C_out, C_in/groups, k, k, k]
        self.weight = nn.Parameter(
            torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size, kernel_size),
            requires_grad=True
        )
        # Optional per-expert bias: [K, C_out]
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None

    def update_temperature(self):
        """Helper that forwards to attention's scheduler (name unchanged)."""
        self.attention.updata_temperature()

    def forward(self, x):
        """
        x: [B, C_in, D, H, W]
        Return: [B, C_out, D', H', W']
        """
        softmax_attention = self.attention(x)  # [B, K]

        batch_size, in_planes, depth, height, width = x.size()

        # Grouped conv trick: fold batch into groups dimension
        x = x.view(1, -1, depth, height, width)  # [1, B*C_in, D, H, W]

        # Flatten candidate weights over all conv dims for matmul with attention
        weight = self.weight.view(self.K, -1)  # [K, C_out * (C_in/groups) * k^3]

        # Aggregate weights per-sample, then reshape to grouped conv shape
        aggregate_weight = torch.mm(softmax_attention, weight).view(
            batch_size * self.out_planes,
            self.in_planes // self.groups,
            self.kernel_size, self.kernel_size, self.kernel_size
        )

        # Aggregate biases similarly (if present)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)  # [B*C_out]
            output = F.conv3d(
                x, weight=aggregate_weight, bias=aggregate_bias,
                stride=self.stride, padding=self.padding, dilation=self.dilation,
                groups=self.groups * batch_size
            )
        else:
            output = F.conv3d(
                x, weight=aggregate_weight, bias=None,
                stride=self.stride, padding=self.padding, dilation=self.dilation,
                groups=self.groups * batch_size
            )

        # Unfold batch dim back
        output = output.view(batch_size, self.out_planes, output.size(-3), output.size(-2), output.size(-1))
        return output


# ---------------------------------------------------------------------
# Attention gate for UNet skip connections (3D)
# ---------------------------------------------------------------------
class Attention_block(nn.Module):
    """
    Attention gate (3D) that modulates encoder skip features x
    using decoder gating features g. Output is x * alpha, alpha in [0,1].
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        # Project gating feature g and skip feature x to a common F_int
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.InstanceNorm3d(F_int)  # original commented
            nn.GroupNorm(num_groups=8, num_channels=F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.InstanceNorm3d(F_int)
            nn.GroupNorm(num_groups=8, num_channels=F_int)
        )

        # 1x1x1 conv to single-channel attention map, followed by normalization
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.InstanceNorm3d(1),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Softmax()  # kept exactly as in your code to preserve behavior
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: decoder gating feature [B, F_g, D, H, W]
        x: encoder skip feature  [B, F_l, D, H, W]
        returns: gated skip feature same shape as x
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # attention mask (kept as-is)
        return x * psi


# ---------------------------------------------------------------------
# Residual + encoder/decoder building blocks
# ---------------------------------------------------------------------
class ResDoubleConv(nn.Module):
    """BN (GroupNorm) -> LeakyReLU -> DynConv3d -> BN -> LeakyReLU -> DynConv3d, with residual skip."""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.LeakyReLU(inplace=True),
            Dynamic_conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            Dynamic_conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.skip = nn.Sequential(
            Dynamic_conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        )

    def forward(self, x):
        return self.double_conv(x) + self.skip(x)


class ResDown(nn.Module):
    """Encoder step: 3D MaxPool(2) followed by ResDoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            ResDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class Out(nn.Module):
    """Final projection via 1x1x1 dynamic conv to produce class logits."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Dynamic_conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    """
    Decoder upsampling step using transposed 3D convolution.
    Note: signature and attribute names kept intact for weight compatibility.
    """
    def __init__(self, in_channels, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, in_channels // 2,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x):
        x = self.up(x)
        return x


# ---------------------------------------------------------------------
# Full UNet with attention skips and dynamic convolutions
# ---------------------------------------------------------------------
class DSNet(nn.Module):
    """
    UNet-style encoder/decoder:
      - Input stem with residual connection
      - 3 x encoder downs + bridge
      - 4 x decoder ups with attention-gated skips
      - Final 1x1x1 projection to n_classes
    """
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        # Input stem: two dynamic convs + skip projection
        self.input_layer = nn.Sequential(
            Dynamic_conv3d(in_channels, n_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=n_channels),
            nn.LeakyReLU(inplace=True),
            Dynamic_conv3d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        )
        self.input_skip = Dynamic_conv3d(in_channels, n_channels, kernel_size=3, stride=1, padding=1)

        # Encoder path
        self.enc1 = ResDown(n_channels, 2 * n_channels)
        self.enc2 = ResDown(2 * n_channels, 4 * n_channels)
        self.enc3 = ResDown(4 * n_channels, 8 * n_channels)
        self.bridge = ResDown(8 * n_channels, 16 * n_channels)

        # Decoder path with attention-gated skips
        self.Up4 = up_conv(16 * n_channels, 8 * n_channels)
        self.Att4 = Attention_block(F_g=8 * n_channels, F_l=8 * n_channels, F_int=4 * n_channels)
        self.Up_conv4 = ResDoubleConv(16 * n_channels, 8 * n_channels)

        self.Up3 = up_conv(8 * n_channels, 4 * n_channels)
        self.Att3 = Attention_block(F_g=4 * n_channels, F_l=4 * n_channels, F_int=2 * n_channels)
        self.Up_conv3 = ResDoubleConv(8 * n_channels, 4 * n_channels)

        self.Up2 = up_conv(4 * n_channels, 2 * n_channels)
        self.Att2 = Attention_block(F_g=2 * n_channels, F_l=2 * n_channels, F_int=1 * n_channels)
        self.Up_conv2 = ResDoubleConv(4 * n_channels, 2 * n_channels)

        self.Up1 = up_conv(2 * n_channels, 1 * n_channels)
        self.Att1 = Attention_block(F_g=1 * n_channels, F_l=1 * n_channels, F_int=n_channels // 2)
        self.Up_conv1 = ResDoubleConv(2 * n_channels, 1 * n_channels)

        # Final logits
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        # Input stem with residual (x -> n_channels)
        x1 = self.input_layer(x) + self.input_skip(x)

        # Encoder downs
        x2 = self.enc1(x1)  # n -> 2n
        x3 = self.enc2(x2)  # 2n -> 4n
        x4 = self.enc3(x3)  # 4n -> 8n
        bridge = self.bridge(x4)  # 8n -> 16n

        # Decoder + attention-gated skips
        d4 = self.Up4(bridge)
        x4 = self.Att4(g=d4, x=x4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x3 = self.Att3(g=d3, x=x3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x2 = self.Att2(g=d2, x=x2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        x1 = self.Att1(g=d1, x=x1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Up_conv1(d1)

        # Head
        mask = self.out(d1)  # logits: [B, n_classes, D, H, W]
        return mask
    
    
# -----------------------------------------------------------------------------

