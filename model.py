"""WSRNet — Wavelet Scale-Recurrent Network for pansharpening.

The network treats the wavelet pyramid as a coarse-to-fine sequence and fuses
PAN/MS across scales with a single weight-shared recurrent state cell
(ConvGRU + dw-window-attention gates), reconstructing each level by IDWT.
The hidden state h is deterministic (no stochastic latent z).

The main model class is RSSMHWViTHZ (kept for checkpoint compatibility); use
the WSRNet alias at the bottom of this file as the public entry point. The
WFANet baseline lives in net_torch.HWViT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from net_torch import DWT_2D, IDWT_2D, raise_channel, reduce_channel, resblock, DWC, FFN, FFN_2

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception:
    Mamba = None

try:
    from torchvision.ops import DeformConv2d
except Exception:
    DeformConv2d = None


class WaveletPyramid(nn.Module):
    def __init__(self, levels=3):
        super().__init__()
        self.levels = levels
        self.dwt = DWT_2D()

    def forward(self, x):
        coeffs = []
        current = x
        for _ in range(self.levels):
            dec = self.dwt(current)
            c = dec.shape[1] // 4
            ll = dec[:, :c]
            lh = dec[:, c: 2 * c]
            hl = dec[:, 2 * c: 3 * c]
            hh = dec[:, 3 * c:]
            coeffs.append((ll, lh, hl, hh))
            current = ll
        return coeffs


class WaveletReconstructor(nn.Module):
    def __init__(self, levels=3):
        super().__init__()
        self.levels = levels
        self.idwt = IDWT_2D()

    def forward(self, coeffs):
        current = coeffs[-1][0]
        for i in range(self.levels - 1, -1, -1):
            ll, lh, hl, hh = coeffs[i]
            if i == self.levels - 1:
                pack = torch.cat([ll, lh, hl, hh], dim=1)
            else:
                pack = torch.cat([current, lh, hl, hh], dim=1)
            current = self.idwt(pack)
        return current


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW tensors."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class MultiDilatedGateConv2d(nn.Module):
    """Mixed-dilation depthwise conv followed by pointwise fusion.

    The final 3x3 clean-up convolution is intentionally standard to reduce
    gridding artifacts from the dilated branches.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(in_channels, in_channels, 3, 1, dilation, dilation=dilation,
                          groups=in_channels, bias=True)
                for dilation in (1, 2, 3)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, 1, 0, bias=True),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class DeformableGateConv2d(nn.Module):
    """Deformable convolution operator for ConvGRU gates/candidate.

    Offsets are zero-initialized, so the module starts close to a standard
    convolution and can learn spatially adaptive sampling only if useful.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        if DeformConv2d is None:
            raise ImportError("torchvision.ops.DeformConv2d is required for state_conv_type='deformable'")
        kernel_size = int(kernel_size)
        padding = kernel_size // 2
        self.offset = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size,
            1,
            padding,
            bias=True,
        )
        self.conv = DeformConv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=True)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x):
        return self.conv(x, self.offset(x))


class WindowAttentionGateConv2d(nn.Module):
    """Local window-attention residual operator for ConvGRU gates/candidate.

    This keeps the baseline convolution path and adds a zero-initialized
    window-attention residual, preserving the O(N * window^2) local-compute
    story without jumping to full O(N^2) attention.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        window_size=8,
        heads=4,
        local_type="plain",
        shift_size=0,
        use_relative_bias=False,
    ):
        super().__init__()
        kernel_size = int(kernel_size)
        padding = kernel_size // 2
        if local_type == "dw_large":
            self.local = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, 1, padding,
                          groups=in_channels, bias=True),
                nn.PReLU(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True),
            )
        else:
            self.local = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=True)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.use_relative_bias = bool(use_relative_bias)
        self.heads = self._choose_heads(out_channels, heads)
        self.head_dim = out_channels // self.heads
        self.q = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.k = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.v = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.proj = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.use_relative_bias:
            size = (2 * self.window_size - 1) * (2 * self.window_size - 1)
            self.relative_position_bias_table = nn.Parameter(torch.zeros(size, self.heads))
            self.register_buffer(
                "relative_position_index",
                self._build_relative_position_index(self.window_size),
                persistent=False,
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        else:
            self.relative_position_bias_table = None
            self.register_buffer("relative_position_index", torch.empty(0, dtype=torch.long), persistent=False)

    @staticmethod
    def _choose_heads(channels, max_heads):
        max_heads = max(1, min(int(max_heads), int(channels)))
        for h in range(max_heads, 0, -1):
            if channels % h == 0:
                return h
        return 1

    @staticmethod
    def _build_relative_position_index(window_size):
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        return relative_coords.sum(-1).long()

    def _window_partition(self, x):
        b, c, h, w = x.shape
        ws = max(1, min(self.window_size, max(h, w)))
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        hp, wp = x.shape[-2:]
        x = x.view(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(b * (hp // ws) * (wp // ws), ws * ws, c)
        return x, (b, c, h, w, hp, wp, ws)

    def _window_unpartition(self, x, meta):
        b, c, h, w, hp, wp, ws = meta
        x = x.view(b, hp // ws, wp // ws, ws, ws, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(b, c, hp, wp)
        return x[:, :, :h, :w]

    def _shifted_window_mask(self, meta, shift, device):
        b, c, h, w, hp, wp, ws = meta
        if shift <= 0 or hp <= ws or wp <= ws:
            return None
        img_mask = torch.zeros((1, 1, hp, wp), device=device)
        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, :, h_slice, w_slice] = cnt
                cnt += 1
        mask_windows, _ = self._window_partition(img_mask)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def _relative_bias(self, tokens, ws, device):
        if (
            not self.use_relative_bias
            or self.relative_position_bias_table is None
            or ws != self.window_size
            or tokens != ws * ws
        ):
            return None
        index = self.relative_position_index.to(device)
        bias = self.relative_position_bias_table[index.reshape(-1)]
        bias = bias.reshape(tokens, tokens, -1).permute(2, 0, 1).contiguous()
        return bias.unsqueeze(0)

    def forward(self, x):
        local = self.local(x)
        _, _, h, w = x.shape
        shift = min(self.shift_size, self.window_size // 2)
        if shift > 0 and h > self.window_size and w > self.window_size:
            x_attn = torch.roll(x, shifts=(-shift, -shift), dims=(2, 3))
        else:
            shift = 0
            x_attn = x
        q, meta = self._window_partition(self.q(x_attn))
        k, _ = self._window_partition(self.k(x_attn))
        v, _ = self._window_partition(self.v(x_attn))
        b_windows, tokens, channels = q.shape
        q = q.view(b_windows, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b_windows, tokens, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b_windows, tokens, self.heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        rel_bias = self._relative_bias(tokens, meta[-1], attn.device)
        if rel_bias is not None:
            attn = attn + rel_bias
        attn_mask = self._shifted_window_mask(meta, shift, attn.device)
        if attn_mask is not None:
            n_windows = attn_mask.shape[0]
            attn = attn.view(-1, n_windows, self.heads, tokens, tokens)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.heads, tokens, tokens)
        attn = torch.softmax(attn, dim=-1)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b_windows, tokens, channels)
        y = self._window_unpartition(y, meta)
        if shift > 0:
            y = torch.roll(y, shifts=(shift, shift), dims=(2, 3))
        return local + self.gamma * self.proj(y)


def make_gate_conv2d(in_channels, out_channels, kernel_size=3, conv_type="plain"):
    """Build the convolution operator used inside ConvGRU gates."""
    kernel_size = int(kernel_size)
    padding = kernel_size // 2
    conv_type = str(conv_type)
    if conv_type == "plain":
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    if conv_type == "dw_large":
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, 1, padding,
                      groups=in_channels, bias=True),
            nn.PReLU(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True),
        )
    if conv_type == "convnext_dw":
        hidden = max(in_channels, out_channels * 2)
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, 1, padding,
                      groups=in_channels, bias=True),
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, hidden, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 1, 1, 0, bias=True),
        )
    if conv_type == "ms_dilated":
        return MultiDilatedGateConv2d(in_channels, out_channels)
    if conv_type == "deformable":
        return DeformableGateConv2d(in_channels, out_channels, kernel_size)
    if conv_type == "window_attn":
        return WindowAttentionGateConv2d(in_channels, out_channels, kernel_size, window_size=8, heads=4)
    if conv_type == "dw_window_attn":
        return WindowAttentionGateConv2d(
            in_channels,
            out_channels,
            kernel_size,
            window_size=8,
            heads=4,
            local_type="dw_large",
        )
    if conv_type == "swin_window_attn":
        return WindowAttentionGateConv2d(
            in_channels,
            out_channels,
            kernel_size,
            window_size=8,
            heads=4,
            shift_size=4,
            use_relative_bias=True,
        )
    raise ValueError(f"Unsupported state conv type: {conv_type}")


class ConvGRUCell2d(nn.Module):
    """2D convolutional GRU cell that preserves spatial structure during state updates."""
    def __init__(self, input_dim, hidden_dim, kernel_size=3, conv_type="plain"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reset = make_gate_conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, conv_type)
        self.update = make_gate_conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, conv_type)
        self.candidate = make_gate_conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, conv_type)

    def forward(self, x, h_prev, gate_mod=None):
        combined = torch.cat([x, h_prev], dim=1)
        reset_logits = self.reset(combined)
        update_logits = self.update(combined)
        if gate_mod is not None:
            r_mod, u_mod = torch.chunk(gate_mod, 2, dim=1)
            reset_logits = reset_logits + r_mod
            update_logits = update_logits + u_mod
        r = torch.sigmoid(reset_logits)
        u = torch.sigmoid(update_logits)
        n = torch.tanh(self.candidate(torch.cat([x, r * h_prev], dim=1)))
        return (1.0 - u) * h_prev + u * n


class ZeroInitResidual(nn.Module):
    """Small residual block whose last layer starts from zero output."""
    def __init__(self, channels, hidden_channels=None, depthwise=True):
        super().__init__()
        hidden_channels = hidden_channels or channels
        groups = channels if depthwise else 1
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=groups, bias=True),
            nn.PReLU(channels),
            nn.Conv2d(channels, hidden_channels, 1, 1, 0, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return x + self.body(x)


class StateSpatialMixer(nn.Module):
    """Local spatial context mixer for recurrent hidden states."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mix = ZeroInitResidual(hidden_dim, hidden_channels=hidden_dim, depthwise=True)

    def forward(self, h_state):
        return self.mix(h_state)


class RSSMHzCell(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim,
        latent_dim,
        deterministic_only=False,
        use_conv_gru=False,
        state_conv_type="plain",
        state_kernel_size=3,
        z_eval_mode="prior",
        z_update_order="legacy",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.deterministic_only = deterministic_only
        self.use_conv_gru = use_conv_gru
        self.z_eval_mode = z_eval_mode
        self.z_update_order = z_update_order

        if use_conv_gru:
            self.gru = ConvGRUCell2d(
                obs_dim + latent_dim,
                hidden_dim,
                kernel_size=state_kernel_size,
                conv_type=state_conv_type,
            )
            self.prior = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, latent_dim * 2, 1),
            )
            self.posterior = nn.Sequential(
                nn.Conv2d(hidden_dim + obs_dim, hidden_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, latent_dim * 2, 1),
            )
        else:
            self.gru = nn.GRUCell(obs_dim + latent_dim, hidden_dim)
            self.prior = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim * 2),
            )
            self.posterior = nn.Sequential(
                nn.Linear(hidden_dim + obs_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim * 2),
            )

    @staticmethod
    def _sample(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return torch.clamp(z, min=-6.0, max=6.0)

    @staticmethod
    def _kl_div(mu_q, logvar_q, mu_p, logvar_p):
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8) - 1.0)
        # Sum over channel dim (dim=1) for 2D, same for 1D
        return kl.sum(dim=1)

    def set_z_eval_mode(self, mode):
        if mode not in {"prior", "posterior", "zero"}:
            raise ValueError(f"Unsupported z_eval_mode: {mode}")
        self.z_eval_mode = mode

    def _latent_from_state(self, h_ref, obs, z_like, training, force_zero_z=False):
        if self.deterministic_only or force_zero_z:
            z = torch.zeros_like(z_like)
            kl = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
            return z, kl

        prior_stats = self.prior(h_ref)
        mu_p, logvar_p = torch.chunk(prior_stats, 2, dim=1)
        logvar_p = torch.clamp(logvar_p, min=-8.0, max=2.0)

        need_posterior = training or self.z_eval_mode == "posterior"
        if need_posterior:
            post_stats = self.posterior(torch.cat([h_ref, obs], dim=1))
            mu_q, logvar_q = torch.chunk(post_stats, 2, dim=1)
            logvar_q = torch.clamp(logvar_q, min=-8.0, max=2.0)
            kl = self._kl_div(mu_q, logvar_q, mu_p, logvar_p)
            z = self._sample(mu_q, logvar_q) if training else mu_q
        elif self.z_eval_mode == "zero":
            z = torch.zeros_like(mu_p)
            kl = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        else:
            z = mu_p
            kl = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        return z, kl

    def _forward_2d(self, obs, h_prev, z_prev, training, force_zero_z=False, gate_mod=None):
        if self.z_update_order == "innovation":
            z, kl = self._latent_from_state(h_prev, obs, z_prev, training, force_zero_z=force_zero_z)
            h_state = self.gru(torch.cat([obs, z], dim=1), h_prev, gate_mod=gate_mod)
            return h_state, z, kl

        h_bar = self.gru(torch.cat([obs, z_prev], dim=1), h_prev, gate_mod=gate_mod)
        z, kl = self._latent_from_state(h_bar, obs, z_prev, training, force_zero_z=force_zero_z)
        return h_bar, z, kl

    def _forward_1d(self, obs, h_prev, z_prev, training, force_zero_z=False):
        if self.z_update_order == "innovation":
            z, kl = self._latent_from_state(h_prev, obs, z_prev, training, force_zero_z=force_zero_z)
            h_state = self.gru(torch.cat([obs, z], dim=1), h_prev)
            return h_state, z, kl

        h_bar = self.gru(torch.cat([obs, z_prev], dim=1), h_prev)
        z, kl = self._latent_from_state(h_bar, obs, z_prev, training, force_zero_z=force_zero_z)
        return h_bar, z, kl

    def forward(self, obs, h_prev, z_prev, training=True, force_zero_z=False, gate_mod=None):
        if self.use_conv_gru:
            return self._forward_2d(obs, h_prev, z_prev, training, force_zero_z=force_zero_z, gate_mod=gate_mod)
        else:
            return self._forward_1d(obs, h_prev, z_prev, training, force_zero_z=force_zero_z)


class LevelLLCorrection(nn.Module):
    """Level-wise LL correction before IDWT reconstruction.

    This is more targeted than the final-image low-frequency head: it lets each
    wavelet level correct LL/spectral bias while preserving the MS_LL residual
    path at initialization.
    """
    def __init__(self, pan_channels, ms_channels):
        super().__init__()
        hidden = ms_channels
        self.body = nn.Sequential(
            nn.Conv2d(ms_channels * 2 + pan_channels, hidden, 1, 1, 0, bias=True),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=True),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, ms_channels, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)

    def forward(self, fused_ll, ms_ll, pan_feat):
        delta = self.body(torch.cat([fused_ll, ms_ll, pan_feat], dim=1))
        return fused_ll + delta


class CrossScaleFusionHz(nn.Module):
    def __init__(
        self,
        pan_channels,
        ms_channels,
        hidden_dim,
        latent_dim,
        deterministic_only=False,
        use_conv_gru=False,
        state_conv_type="plain",
        state_kernel_size=3,
        use_freq_gated_state=False,
        gate_mod_channels=None,
        use_state_spatial_mixer=False,
        use_level_ll_corr=False,
        use_local_freq_mixer=False,
        lfm_kernel_size=3,
        lfm_hidden_scale=1.0,
        use_linear_freq_attention=False,
        linear_attn_heads=4,
        z_eval_mode="prior",
        z_update_order="legacy",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.deterministic_only = deterministic_only
        self.use_conv_gru = use_conv_gru
        self.use_freq_gated_state = bool(use_freq_gated_state)
        self.pan_state_channels = pan_channels // 4 if self.use_freq_gated_state else pan_channels

        self.pan_proj = nn.Sequential(
            nn.Conv2d(self.pan_state_channels, hidden_dim // 2, 1, bias=True),
            nn.PReLU(hidden_dim // 2),
        )
        self.ms_proj = nn.Sequential(
            nn.Conv2d(ms_channels, hidden_dim // 2, 1, bias=True),
            nn.PReLU(hidden_dim // 2),
        )

        self.cell = RSSMHzCell(hidden_dim, hidden_dim, latent_dim,
                               deterministic_only=deterministic_only, use_conv_gru=use_conv_gru,
                               state_conv_type=state_conv_type, state_kernel_size=state_kernel_size,
                               z_eval_mode=z_eval_mode, z_update_order=z_update_order)
        self.state_mixer = StateSpatialMixer(hidden_dim) if use_state_spatial_mixer else None
        if self.use_freq_gated_state:
            gate_mod_channels = gate_mod_channels or (pan_channels - self.pan_state_channels)
            self.freq_gate_mod = nn.Sequential(
                nn.Conv2d(gate_mod_channels, hidden_dim, 1, 1, 0, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim * 2, 1, 1, 0, bias=True),
            )
            nn.init.zeros_(self.freq_gate_mod[-1].weight)
            if self.freq_gate_mod[-1].bias is not None:
                nn.init.zeros_(self.freq_gate_mod[-1].bias)
        else:
            self.freq_gate_mod = None

        self.hz_to_feat = nn.Sequential(
            nn.Conv2d(hidden_dim + latent_dim, hidden_dim, 3, 1, 1, bias=True),
            nn.PReLU(hidden_dim),
            nn.Conv2d(hidden_dim, ms_channels, 3, 1, 1, bias=True),
        )

        self.obs_gate = nn.Sequential(
            nn.Conv2d(hidden_dim, ms_channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.level_ll_corr = LevelLLCorrection(self.pan_state_channels, ms_channels) if use_level_ll_corr else None

    def set_z_eval_mode(self, mode):
        self.cell.set_z_eval_mode(mode)

    def _decode_ll(self, h_state, z_state, obs, ms_feat, pan_feat):
        fused_raw = self.hz_to_feat(torch.cat([h_state, z_state], dim=1))
        gate = self.obs_gate(obs)
        fused = fused_raw * gate + ms_feat
        if self.level_ll_corr is not None:
            fused = self.level_ll_corr(fused, ms_feat, pan_feat)
        return fused

    def forward(
        self,
        pan_feat,
        ms_feat,
        h_prev,
        z_prev,
        training=True,
        force_zero_z=False,
        return_z0=False,
        gate_mod_feat=None,
    ):
        pan_state_feat = pan_feat
        gate_mod = None
        if self.use_freq_gated_state:
            c = pan_feat.shape[1] // 4
            pan_state_feat = pan_feat[:, :c]
            if gate_mod_feat is None:
                gate_mod_feat = pan_feat[:, c:]
            gate_mod = self.freq_gate_mod(gate_mod_feat)

        p = self.pan_proj(pan_state_feat)
        m = self.ms_proj(ms_feat)
        obs = torch.cat([p, m], dim=1)

        if self.use_conv_gru:
            h_state, z_state, kl = self.cell(
                obs,
                h_prev,
                z_prev,
                training=training,
                force_zero_z=force_zero_z,
                gate_mod=gate_mod,
            )
        else:
            b, _, h, w = pan_feat.shape
            obs_flat = obs.permute(0, 2, 3, 1).reshape(b * h * w, self.hidden_dim)
            h_prev_flat = h_prev.permute(0, 2, 3, 1).reshape(b * h * w, self.hidden_dim)
            z_prev_flat = z_prev.permute(0, 2, 3, 1).reshape(b * h * w, self.latent_dim)
            h_flat, z_flat, kl = self.cell(
                obs_flat,
                h_prev_flat,
                z_prev_flat,
                training=training,
                force_zero_z=force_zero_z,
            )
            h_state = h_flat.reshape(b, h, w, self.hidden_dim).permute(0, 3, 1, 2)
            z_state = z_flat.reshape(b, h, w, self.latent_dim).permute(0, 3, 1, 2)

        if self.state_mixer is not None:
            h_state = self.state_mixer(h_state)

        fused = self._decode_ll(h_state, z_state, obs, ms_feat, pan_state_feat)
        fused_z0 = None
        if return_z0:
            fused_z0 = self._decode_ll(h_state, torch.zeros_like(z_state), obs, ms_feat, pan_state_feat)

        kl_mean = kl.mean() if kl.numel() > 0 else kl
        return fused, h_state, z_state, kl_mean, fused_z0


class LowFreqCorrection(nn.Module):
    """Lightweight low-frequency/spectral correction head.

    The last conv is zero-initialized, so enabling this module starts from the
    original RSSM-HZ output and learns only a residual correction when useful.
    """
    def __init__(self, channels, hidden_channels=32, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size
        self.body = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, hidden_channels, 3, 1, 1, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, channels, 3, 1, 1, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, fused, ms_up, pan):
        pad = self.kernel_size // 2
        fused_lf = F.avg_pool2d(fused, self.kernel_size, stride=1, padding=pad)
        ms_lf = F.avg_pool2d(ms_up, self.kernel_size, stride=1, padding=pad)
        pan_lf = F.avg_pool2d(pan, self.kernel_size, stride=1, padding=pad)
        return self.gamma * self.body(torch.cat([fused_lf, ms_lf, pan_lf], dim=1))


class BandAwareCorrection(nn.Module):
    """Per-band spectral residual correction with local and global cues.

    GF2/QB error analysis showed that the remaining error is concentrated in a
    few multispectral bands. This head is deliberately small and zero-initialized:
    it starts as an exact no-op and learns only a residual correction on top of
    the existing RSSM-HZ output. Its cost is linear in the number of pixels.
    """
    def __init__(self, channels, hidden_channels=32, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        in_channels = channels * 3 + 1
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, padding, groups=hidden_channels, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=True),
        )
        self.global_affine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, channels * 2, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)
        nn.init.zeros_(self.global_affine[-1].weight)
        nn.init.zeros_(self.global_affine[-1].bias)

    def forward(self, base, fused, ms_up, pan):
        x = torch.cat([base, fused, ms_up, pan], dim=1)
        local_delta = self.local(x)
        stats = F.adaptive_avg_pool2d(x, 1)
        gamma, beta = torch.chunk(self.global_affine(stats), 2, dim=1)
        return local_delta + gamma * ms_up + beta


class SDEMLite(nn.Module):
    """Lightweight spatial detail enhancement branch.

    PAN feature details are decomposed/recomposed in wavelet space and added as
    a zero-initialized residual before channel reduction. It approximates the
    role of WFANet's SDEM without introducing quadratic attention.
    """
    def __init__(self, channels):
        super().__init__()
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        self.band_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=True),
                    nn.PReLU(channels),
                    nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
                    nn.Sigmoid(),
                )
                for _ in range(4)
            ]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.proj[-1].weight)
        if self.proj[-1].bias is not None:
            nn.init.zeros_(self.proj[-1].bias)

    def forward(self, pan_feat):
        coeffs = self.dwt(pan_feat)
        c = pan_feat.shape[1]
        bands = [
            coeffs[:, :c],
            coeffs[:, c: 2 * c],
            coeffs[:, 2 * c: 3 * c],
            coeffs[:, 3 * c:],
        ]
        enhanced = [band * self.band_gates[i](band) for i, band in enumerate(bands)]
        detail = self.idwt(torch.cat(enhanced, dim=1))
        return self.proj(detail)


class WFANetWeightedCombine(nn.Module):
    """WFANet-style learnable weighted sum followed by local residual refinement."""
    def __init__(self, channels):
        super().__init__()
        self.resblock = resblock(channel=channels)
        self.a = nn.Parameter(torch.tensor(0.33))
        self.b = nn.Parameter(torch.tensor(0.33))

    def forward(self, x1, x2, x3):
        target_size = x2.shape[-2:]
        if x1.shape[-2:] != target_size:
            x1 = F.interpolate(x1, size=target_size, mode="bilinear", align_corners=False)
        if x3.shape[-2:] != target_size:
            x3 = F.interpolate(x3, size=target_size, mode="bilinear", align_corners=False)
        mixed = self.a * x1 + self.b * x2 + (1.0 - self.a - self.b) * x3
        return self.resblock(mixed)


class WFANetPanDetailPath(nn.Module):
    """Official-WFANet-inspired PAN detail path: subband DWC gates + IDWT + conv."""
    def __init__(self, pan_channels, out_channels):
        super().__init__()
        self.idwt = IDWT_2D()
        self.wd_ll_conv = DWC(channel=pan_channels)
        self.wd_lh_conv = DWC(channel=pan_channels)
        self.wd_hl_conv = DWC(channel=pan_channels)
        self.wd_hh_conv = DWC(channel=pan_channels)
        self.detail_res = resblock(channel=pan_channels)
        hidden = max(1, out_channels // 2)
        self.to_ms = FFN_2(in_channel=pan_channels, FFN_channel=hidden, out_channel=out_channels)

    def forward(self, wd_ll, wd_lh, wd_hl, wd_hh):
        packed = torch.cat(
            [
                self.wd_ll_conv(wd_ll),
                self.wd_lh_conv(wd_lh),
                self.wd_hl_conv(wd_hl),
                self.wd_hh_conv(wd_hh),
            ],
            dim=1,
        )
        detail = self.detail_res(self.idwt(packed))
        return self.to_ms(detail)


class WFANetStyleRSSMStage(nn.Module):
    """One WFANet-like DWT/IDWT stage with RSSM replacing quadratic attention.

    The stage follows the official WFANet dataflow:
    PAN feature -> DWT subbands, a value branch from PAN_LL/LMS/back_img, subband
    fusion, IDWT reconstruction, plus a parallel PAN-detail IDWT branch.
    """
    def __init__(
        self,
        pan_channels,
        ms_channels,
        hidden_dim,
        latent_dim,
        deterministic_only=False,
        separate_subband_gates=True,
        use_conv_gru=False,
        state_conv_type="plain",
        state_kernel_size=3,
        use_freq_gated_state=False,
        signed_hf_gate=False,
        hf_gate_scale=1.0,
        use_state_spatial_mixer=False,
        use_level_ll_corr=False,
        z_eval_mode="prior",
        z_update_order="legacy",
        use_local_freq_mixer=False,
        lfm_kernel_size=3,
        lfm_hidden_scale=1.0,
        use_linear_freq_attention=False,
        linear_attn_heads=4,
    ):
        super().__init__()
        self.pan_channels = pan_channels
        self.ms_channels = ms_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.separate_subband_gates = separate_subband_gates
        self.signed_hf_gate = signed_hf_gate
        self.hf_gate_scale = hf_gate_scale
        self.use_local_freq_mixer = use_local_freq_mixer
        self.use_linear_freq_attention = use_linear_freq_attention
        self.use_freq_gated_state = bool(use_freq_gated_state)

        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        self.pan_ll_to_ms = (
            nn.Conv2d(pan_channels, ms_channels, 1, bias=True)
            if pan_channels != ms_channels else nn.Identity()
        )
        self.back_mlp = FFN(in_channel=ms_channels, FFN_channel=max(1, ms_channels // 2), out_channel=ms_channels)
        self.value_combine = WFANetWeightedCombine(ms_channels)
        self.value_refine = resblock(channel=ms_channels)

        self.state_fusion = CrossScaleFusionHz(
            pan_channels=pan_channels * 4,
            ms_channels=ms_channels,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            deterministic_only=deterministic_only,
            use_conv_gru=use_conv_gru,
            state_conv_type=state_conv_type,
            state_kernel_size=state_kernel_size,
            use_freq_gated_state=use_freq_gated_state,
            gate_mod_channels=pan_channels * 3 + ms_channels * 3,
            use_state_spatial_mixer=use_state_spatial_mixer,
            use_level_ll_corr=use_level_ll_corr,
            z_eval_mode=z_eval_mode,
            z_update_order=z_update_order,
        )

        self.pan_high_to_ms = nn.ModuleDict(
            {
                "lh": nn.Conv2d(pan_channels, ms_channels, 1, bias=True),
                "hl": nn.Conv2d(pan_channels, ms_channels, 1, bias=True),
                "hh": nn.Conv2d(pan_channels, ms_channels, 1, bias=True),
            }
        )
        self.z_to_gate = nn.Conv2d(latent_dim, ms_channels, 1, bias=True)
        gate_in_channels = ms_channels * 6
        if separate_subband_gates:
            self.high_gate_lh = nn.Sequential(nn.Conv2d(gate_in_channels, ms_channels, 1, bias=True), nn.Sigmoid())
            self.high_gate_hl = nn.Sequential(nn.Conv2d(gate_in_channels, ms_channels, 1, bias=True), nn.Sigmoid())
            self.high_gate_hh = nn.Sequential(nn.Conv2d(gate_in_channels, ms_channels, 1, bias=True), nn.Sigmoid())
        else:
            self.high_gate = nn.Sequential(nn.Conv2d(gate_in_channels, ms_channels, 1, bias=True), nn.Sigmoid())

        self.pan_detail = WFANetPanDetailPath(pan_channels, ms_channels)
        if use_local_freq_mixer:
            self.local_mixer_lh = LocalFrequencyMixer(ms_channels, lfm_kernel_size, lfm_hidden_scale)
            self.local_mixer_hl = LocalFrequencyMixer(ms_channels, lfm_kernel_size, lfm_hidden_scale)
            self.local_mixer_hh = LocalFrequencyMixer(ms_channels, lfm_kernel_size, lfm_hidden_scale)
        if use_linear_freq_attention:
            self.linear_attn_lh = LinearFrequencyAttention(ms_channels, linear_attn_heads)
            self.linear_attn_hl = LinearFrequencyAttention(ms_channels, linear_attn_heads)
            self.linear_attn_hh = LinearFrequencyAttention(ms_channels, linear_attn_heads)
        self.conv_v = FFN_2(in_channel=ms_channels, FFN_channel=max(1, ms_channels // 2), out_channel=ms_channels)
        self.out_refine = resblock(channel=ms_channels)

    def _hf_alpha(self, alpha):
        if self.signed_hf_gate:
            return alpha + self.hf_gate_scale * (2.0 * alpha - 1.0)
        return self.hf_gate_scale * alpha

    def set_z_eval_mode(self, mode):
        self.state_fusion.set_z_eval_mode(mode)

    def forward(self, pan_feat, l_up, back_img, h_prev, z_prev, training=True):
        pan_w = self.dwt(pan_feat)
        c = self.pan_channels
        wd_ll = pan_w[:, :c]
        wd_lh = pan_w[:, c:2 * c]
        wd_hl = pan_w[:, 2 * c:3 * c]
        wd_hh = pan_w[:, 3 * c:]

        l_up = F.interpolate(l_up, size=wd_ll.shape[-2:], mode="bilinear", align_corners=False)
        back_img = F.interpolate(back_img, size=wd_ll.shape[-2:], mode="bilinear", align_corners=False)
        pan_ll_ms = self.pan_ll_to_ms(wd_ll)
        value = self.value_combine(pan_ll_ms, l_up, self.back_mlp(back_img))
        value = self.value_refine(value)

        b, _, h, w = wd_ll.shape
        if h_prev is None:
            h_prev = torch.zeros(b, self.hidden_dim, h, w, device=wd_ll.device, dtype=wd_ll.dtype)
        if z_prev is None:
            z_prev = torch.zeros(b, self.latent_dim, h, w, device=wd_ll.device, dtype=wd_ll.dtype)
        if h_prev.shape[-2:] != (h, w):
            h_prev = F.interpolate(h_prev, size=(h, w), mode="bilinear", align_corners=False)
        if z_prev.shape[-2:] != (h, w):
            z_prev = F.interpolate(z_prev, size=(h, w), mode="bilinear", align_corners=False)

        pan_state = torch.cat([wd_ll, wd_lh, wd_hl, wd_hh], dim=1)
        gate_mod_feat = torch.cat([wd_lh, wd_hl, wd_hh, value, value, value], dim=1) if self.use_freq_gated_state else None
        fused_ll, h_state, z_state, kl_mean, _ = self.state_fusion(
            pan_state,
            value,
            h_prev,
            z_prev,
            training=training,
            force_zero_z=False,
            return_z0=False,
            gate_mod_feat=gate_mod_feat,
        )

        pan_lh = self.pan_high_to_ms["lh"](wd_lh)
        pan_hl = self.pan_high_to_ms["hl"](wd_hl)
        pan_hh = self.pan_high_to_ms["hh"](wd_hh)
        z_gate = self.z_to_gate(z_state)
        gate_in = torch.cat([fused_ll, value, pan_lh, pan_hl, pan_hh, z_gate], dim=1)
        if self.separate_subband_gates:
            alpha_lh = self._hf_alpha(self.high_gate_lh(gate_in))
            alpha_hl = self._hf_alpha(self.high_gate_hl(gate_in))
            alpha_hh = self._hf_alpha(self.high_gate_hh(gate_in))
        else:
            alpha = self._hf_alpha(self.high_gate(gate_in))
            alpha_lh = alpha_hl = alpha_hh = alpha

        # Like WFANet's per-subband attention outputs, each high band is rebuilt
        # from the shared value feature plus PAN directional detail.
        fused_lh = value + alpha_lh * pan_lh
        fused_hl = value + alpha_hl * pan_hl
        fused_hh = value + alpha_hh * pan_hh
        if self.use_local_freq_mixer:
            fused_lh = fused_lh + self.local_mixer_lh(value, pan_lh, alpha_lh, fused_ll, value)
            fused_hl = fused_hl + self.local_mixer_hl(value, pan_hl, alpha_hl, fused_ll, value)
            fused_hh = fused_hh + self.local_mixer_hh(value, pan_hh, alpha_hh, fused_ll, value)
        if self.use_linear_freq_attention:
            fused_lh = fused_lh + self.linear_attn_lh(pan_lh, pan_ll_ms, value)
            fused_hl = fused_hl + self.linear_attn_hl(pan_hl, pan_ll_ms, value)
            fused_hh = fused_hh + self.linear_attn_hh(pan_hh, pan_ll_ms, value)
        value_idwt = self.idwt(torch.cat([fused_ll, fused_lh, fused_hl, fused_hh], dim=1))

        pan_detail = self.pan_detail(wd_ll, wd_lh, wd_hl, wd_hh)
        out = self.out_refine(pan_detail + self.conv_v(value_idwt))
        return out, h_state, z_state, kl_mean


class WFANetTwoStageRSSMFusion(nn.Module):
    """Two-stage WFANet-style scale schedule with RSSM state fusion.

    Stage 1 mirrors L_MWiT on a downsampled PAN feature and LRMS-scale feature.
    Stage 2 mirrors F_MWiT on the full PAN feature, using stage-1 output as
    coarse context. This avoids the 64x64 -> 8x8 three-level bottleneck.
    """
    def __init__(
        self,
        pan_channels,
        ms_channels,
        hidden_dim=96,
        latent_dim=32,
        deterministic_only=False,
        separate_subband_gates=True,
        use_conv_gru=False,
        state_conv_type="plain",
        state_kernel_size=3,
        use_freq_gated_state=False,
        signed_hf_gate=False,
        hf_gate_scale=1.0,
        use_state_spatial_mixer=False,
        use_level_ll_corr=False,
        use_local_freq_mixer=False,
        lfm_kernel_size=3,
        lfm_hidden_scale=1.0,
        use_linear_freq_attention=False,
        linear_attn_heads=4,
        z_eval_mode="prior",
        z_update_order="legacy",
        share_scale_recurrent=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.separate_subband_gates = separate_subband_gates
        self.learnable_fusion = False
        self.use_local_freq_mixer = use_local_freq_mixer
        self.use_linear_freq_attention = use_linear_freq_attention
        self.use_windowed_freq_mixer = False
        self.use_mamba_freq_mixer = False
        self.share_scale_recurrent = bool(share_scale_recurrent)

        common = dict(
            pan_channels=pan_channels,
            ms_channels=ms_channels,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            deterministic_only=deterministic_only,
            separate_subband_gates=separate_subband_gates,
            use_conv_gru=use_conv_gru,
            state_conv_type=state_conv_type,
            state_kernel_size=state_kernel_size,
            use_freq_gated_state=use_freq_gated_state,
            signed_hf_gate=signed_hf_gate,
            hf_gate_scale=hf_gate_scale,
            use_state_spatial_mixer=use_state_spatial_mixer,
            use_level_ll_corr=use_level_ll_corr,
            use_local_freq_mixer=use_local_freq_mixer,
            lfm_kernel_size=lfm_kernel_size,
            lfm_hidden_scale=lfm_hidden_scale,
            use_linear_freq_attention=use_linear_freq_attention,
            linear_attn_heads=linear_attn_heads,
            z_eval_mode=z_eval_mode,
            z_update_order=z_update_order,
        )
        self.coarse_stage = WFANetStyleRSSMStage(**common)
        self.fine_stage = WFANetStyleRSSMStage(**common)
        if self.share_scale_recurrent:
            # Share only the recurrent LL/state fusion block. Stage-specific
            # value/high-frequency branches remain separate because coarse and
            # fine stages play different reconstruction roles.
            self.fine_stage.state_fusion = self.coarse_stage.state_fusion

        self.pan_down_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.ms_down_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.ms_down_4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.state_up_h = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1, bias=True),
            nn.PReLU(hidden_dim),
        )
        self.state_up_z = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim, 4, 2, 1, bias=True),
            nn.PReLU(latent_dim),
        )
        self.pan_to_ms_full = (
            nn.Conv2d(pan_channels, ms_channels, 1, bias=True)
            if pan_channels != ms_channels else nn.Identity()
        )
        self.fine_mlp = FFN(in_channel=ms_channels, FFN_channel=max(1, ms_channels // 2), out_channel=ms_channels)
        self.final_combine = WFANetWeightedCombine(ms_channels)
        self.final_refine = resblock(channel=ms_channels)

    def set_z_eval_mode(self, mode):
        self.coarse_stage.set_z_eval_mode(mode)
        self.fine_stage.set_z_eval_mode(mode)

    def set_z_diagnostics(self, enabled=True):
        # Diagnostics are intentionally no-op for the two-stage prototype; the
        # old recursive fusion path still supports detailed z statistics.
        self.collect_z_diagnostics = bool(enabled)

    def get_z_diagnostics(self):
        return []

    def forward(self, pan_feat, ms_feat, ms_lr_feat, training=True):
        coarse, h_state, z_state, kl_coarse = self.coarse_stage(
            pan_feat=self.pan_down_2(pan_feat),
            l_up=self.ms_down_4(ms_feat),
            back_img=ms_lr_feat,
            h_prev=None,
            z_prev=None,
            training=training,
        )

        h_state = self.state_up_h(h_state)
        z_state = self.state_up_z(z_state)
        fine, _, _, kl_fine = self.fine_stage(
            pan_feat=pan_feat,
            l_up=self.ms_down_2(ms_feat),
            back_img=coarse,
            h_prev=h_state,
            z_prev=z_state,
            training=training,
        )

        fused = self.final_combine(self.pan_to_ms_full(pan_feat), ms_feat, self.fine_mlp(fine))
        fused = self.final_refine(fused)
        kl_loss = torch.stack([kl_coarse, kl_fine]).mean()
        return fused, kl_loss, None


class ZResidualHead(nn.Module):
    """Predict wavelet subband residuals from [h, z] state at each level.

    Outputs 4 * ms_channels (LL, LH, HL, HH) as zero-initialized residuals.
    Fusion: fused_XX = old_fused_XX + beta_XX * pred_r_XX
    """
    def __init__(self, state_dim, latent_dim, ms_channels):
        super().__init__()
        in_dim = state_dim + latent_dim
        out_dim = ms_channels * 4
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True),
            nn.PReLU(in_dim),
            nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True),
            nn.PReLU(in_dim),
            nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)
        self.beta_ll = nn.Parameter(torch.zeros(1))
        self.beta_hf = nn.Parameter(torch.zeros(1))

    def forward(self, h_state, z_state):
        x = torch.cat([h_state, z_state], dim=1)
        r = self.body(x)
        c = r.shape[1] // 4
        r_ll = r[:, :c]
        r_lh = r[:, c:2*c]
        r_hl = r[:, 2*c:3*c]
        r_hh = r[:, 3*c:]
        return r_ll, r_lh, r_hl, r_hh


class LocalFrequencyMixer(nn.Module):
    """Low-complexity local residual mixer for one high-frequency subband.

    The module starts as an exact no-op because the last projection is
    zero-initialized, but it can learn a local correction from observable
    frequency cues. Complexity is O(HW * C * k^2), not global attention.
    """
    def __init__(self, channels, kernel_size=3, hidden_scale=1.0):
        super().__init__()
        hidden = max(channels, int(round(channels * float(hidden_scale))))
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(channels * 6, hidden, 1, 1, 0, bias=True),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, hidden, kernel_size, 1, padding, groups=hidden, bias=True),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, channels, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, ms_hf, pan_hf, alpha, fused_ll, ms_ll):
        alpha_pan = alpha * pan_hf
        diff = (ms_hf - pan_hf).abs()
        x = torch.cat([ms_hf, pan_hf, alpha_pan, diff, fused_ll, ms_ll], dim=1)
        return self.scale * self.body(x)


class LinearFrequencyAttention(nn.Module):
    """WFANet-style frequency attention with linear spatial complexity.

    WFANet uses PAN frequency bands as attention queries and PAN low-frequency
    structure as keys. This module keeps that physical dataflow but replaces
    quadratic N x N attention with kernelized linear attention O(N * C^2).
    The output is zero-initialized residual correction for safe checkpoint
    warm-starting.
    """
    def __init__(self, channels, heads=4):
        super().__init__()
        heads = max(1, int(heads))
        if channels % heads != 0:
            heads = 1
        self.heads = heads
        self.dim_head = channels // heads
        self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, pan_hf, pan_ll_ms, value):
        b, c, h, w = pan_hf.shape
        n = h * w
        q = self.q_proj(pan_hf).view(b, self.heads, self.dim_head, n).transpose(-1, -2)
        k = self.k_proj(pan_ll_ms).view(b, self.heads, self.dim_head, n).transpose(-1, -2)
        v = self.v_proj(value).view(b, self.heads, self.dim_head, n).transpose(-1, -2)

        q = F.elu(q, inplace=False) + 1.0
        k = F.elu(k, inplace=False) + 1.0
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        k_sum = k.sum(dim=2)
        denom = torch.einsum("bhnd,bhd->bhn", q, k_sum).clamp_min(1e-6).unsqueeze(-1)
        out = torch.einsum("bhnd,bhde->bhne", q, kv) / denom
        out = out.transpose(-1, -2).contiguous().view(b, c, h, w)
        return self.scale * self.out_proj(out)


class ChannelHaarSpectralAdapter(nn.Module):
    """1D channel-wise Haar adapter for the MS residual stream.

    The report argues that MS bands should keep explicit spectral structure.
    This module exposes pairwise low/high spectral responses while starting as
    an exact no-op, so old checkpoints remain behaviorally unchanged at load.
    """
    def __init__(self, channels, hidden_channels=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels * 3 + 1, hidden_channels, 1, 1, 0, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)
        self.scale = nn.Parameter(torch.ones(1))

    @staticmethod
    def channel_haar(x):
        c = x.shape[1]
        if c < 2:
            return x
        if c % 2 != 0:
            # The known WV3/GF2/QB configs are even-channel, but keep a safe
            # fallback so the module is not brittle for custom data.
            x_pair = x[:, :-1]
            tail = x[:, -1:]
        else:
            x_pair = x
            tail = None
        even = x_pair[:, 0::2]
        odd = x_pair[:, 1::2]
        scale = 2.0 ** -0.5
        low = (even + odd) * scale
        high = (even - odd) * scale
        out = torch.cat([low, high], dim=1)
        return torch.cat([out, tail], dim=1) if tail is not None else out

    def forward(self, ms_up, lms, pan):
        spectral = self.channel_haar(ms_up)
        delta = self.body(torch.cat([ms_up, lms, spectral, pan], dim=1))
        return ms_up + self.scale * delta


class WindowedFrequencyMixer(nn.Module):
    """Windowed selective-scan mixer for local frequency correction.

    This is a dependency-free, linear-complexity proxy for the report's
    WSLM/FMamba idea: it scans short local windows in frequency subbands and
    predicts a zero-initialized residual correction.
    """
    def __init__(self, channels, window_size=8, hidden_scale=1.0):
        super().__init__()
        hidden = max(channels, int(round(channels * float(hidden_scale))))
        self.window_size = int(window_size)
        self.in_proj = nn.Conv2d(channels * 6, hidden * 3, 1, 1, 0, bias=True)
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=True)
        self.out_proj = nn.Conv2d(hidden, channels, 1, 1, 0, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        self.scale = nn.Parameter(torch.ones(1))

    def _window_flatten(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (-h) % ws
        pad_w = (-w) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        x = x.view(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(b * (hp // ws) * (wp // ws), c, ws * ws)
        return x, (b, c, h, w, hp, wp)

    def _window_unflatten(self, x, meta):
        b, c, h, w, hp, wp = meta
        ws = self.window_size
        x = x.view(b, hp // ws, wp // ws, c, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, c, hp, wp)
        return x[:, :, :h, :w]

    def _bidirectional_scan(self, value, decay):
        v, meta = self._window_flatten(value)
        d, _ = self._window_flatten(decay)
        d = torch.sigmoid(d).clamp(0.02, 0.98)

        state = torch.zeros(v.shape[0], v.shape[1], device=v.device, dtype=v.dtype)
        forward = []
        for idx in range(v.shape[-1]):
            a = d[:, :, idx]
            state = a * state + (1.0 - a) * v[:, :, idx]
            forward.append(state)
        forward = torch.stack(forward, dim=-1)

        state = torch.zeros_like(state)
        backward = []
        for idx in range(v.shape[-1] - 1, -1, -1):
            a = d[:, :, idx]
            state = a * state + (1.0 - a) * v[:, :, idx]
            backward.append(state)
        backward = torch.stack(backward[::-1], dim=-1)
        return self._window_unflatten(0.5 * (forward + backward), meta)

    def forward(self, ms_hf, pan_hf, alpha, fused_ll, ms_ll):
        alpha_pan = alpha * pan_hf
        diff = (ms_hf - pan_hf).abs()
        x = torch.cat([ms_hf, pan_hf, alpha_pan, diff, fused_ll, ms_ll], dim=1)
        value, gate, decay = torch.chunk(self.in_proj(x), 3, dim=1)
        value = self.dw(value)
        mixed = self._bidirectional_scan(value, decay)
        return self.scale * self.out_proj(mixed * torch.sigmoid(gate))


class MambaFrequencyMixer(nn.Module):
    """True Mamba-based window mixer for high-frequency residual correction.

    The module mirrors WindowedFrequencyMixer's input/output contract but uses
    mamba_ssm's selective scan inside each local window. It is intentionally
    zero-initialized at the output projection, so enabling it starts as a no-op
    residual path and preserves old checkpoints as much as possible.
    """
    def __init__(
        self,
        channels,
        window_size=8,
        hidden_scale=1.0,
        d_state=16,
        d_conv=4,
        expand=2,
        bidirectional=True,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required for --use-mamba-frequency-mixer. "
                "Use the wfanet_mamba environment or disable this flag."
            )
        hidden = max(channels, int(round(channels * float(hidden_scale))))
        self.window_size = int(window_size)
        self.bidirectional = bool(bidirectional)
        self.in_proj = nn.Sequential(
            nn.Conv2d(channels * 6, hidden, 1, 1, 0, bias=True),
            nn.PReLU(hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.mamba = Mamba(
            d_model=hidden,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        self.out_proj = nn.Conv2d(hidden, channels, 1, 1, 0, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        self.scale = nn.Parameter(torch.ones(1))

    def _window_flatten(self, x):
        b, c, h, w = x.shape
        ws = max(1, self.window_size)
        pad_h = (-h) % ws
        pad_w = (-w) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        x = x.view(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(b * (hp // ws) * (wp // ws), ws * ws, c)
        return x, (b, c, h, w, hp, wp)

    def _window_unflatten(self, x, meta):
        b, c, h, w, hp, wp = meta
        ws = max(1, self.window_size)
        x = x.view(b, hp // ws, wp // ws, ws, ws, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(b, c, hp, wp)
        return x[:, :, :h, :w]

    def forward(self, ms_hf, pan_hf, alpha, fused_ll, ms_ll):
        alpha_pan = alpha * pan_hf
        diff = (ms_hf - pan_hf).abs()
        x = torch.cat([ms_hf, pan_hf, alpha_pan, diff, fused_ll, ms_ll], dim=1)
        x = self.in_proj(x)
        seq, meta = self._window_flatten(x)
        seq = self.norm(seq)
        y = self.mamba(seq)
        if self.bidirectional:
            y_rev = torch.flip(self.mamba(torch.flip(seq, dims=[1])), dims=[1])
            y = 0.5 * (y + y_rev)
        mixed = self._window_unflatten(y, meta)
        return self.scale * self.out_proj(mixed)


class RSSMWaveletFusionHz(nn.Module):
    def __init__(self, pan_channels_per_level, ms_channels_per_level, hidden_dim=96, latent_dim=32, levels=3,
                 deterministic_only=False, separate_subband_gates=True, use_conv_gru=False,
                 state_conv_type="plain", state_kernel_size=3, freq_state_mode="mixed",
                 learnable_fusion=False, signed_hf_gate=False, hf_gate_scale=1.0,
                 residual_learnable_fusion=False, use_state_spatial_mixer=False,
                 use_level_ll_corr=False, z_eval_mode="prior", z_update_order="legacy",
                 z_zero_levels=None, use_z_residual_head=False,
                 use_local_freq_mixer=False, lfm_kernel_size=3, lfm_hidden_scale=1.0,
                 use_windowed_freq_mixer=False, wfm_window_size=8, wfm_hidden_scale=1.0,
                 use_mamba_freq_mixer=False, mamba_window_size=8, mamba_hidden_scale=1.0,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 share_scale_recurrent=False):
        super().__init__()
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.deterministic_only = deterministic_only
        self.use_conv_gru = use_conv_gru
        self.state_conv_type = state_conv_type
        self.state_kernel_size = state_kernel_size
        self.freq_state_mode = str(freq_state_mode)
        self.learnable_fusion = learnable_fusion
        self.signed_hf_gate = signed_hf_gate
        self.hf_gate_scale = hf_gate_scale
        self.residual_learnable_fusion = residual_learnable_fusion
        self.use_state_spatial_mixer = use_state_spatial_mixer
        self.use_level_ll_corr = use_level_ll_corr
        self.z_eval_mode = z_eval_mode
        self.z_update_order = z_update_order
        self.z_zero_levels = set(z_zero_levels or [])
        self.collect_z_diagnostics = False
        self.last_z_diagnostics = []
        self.use_z_residual_head = use_z_residual_head
        self.use_local_freq_mixer = use_local_freq_mixer
        self.use_windowed_freq_mixer = use_windowed_freq_mixer
        self.use_mamba_freq_mixer = use_mamba_freq_mixer
        self.share_scale_recurrent = bool(share_scale_recurrent)

        if self.freq_state_mode not in {"mixed", "simple", "split"}:
            raise ValueError(f"Unsupported freq_state_mode: {self.freq_state_mode}")
        if self.share_scale_recurrent and self.freq_state_mode == "split":
            raise ValueError("share_scale_recurrent is for mixed/simple recurrent paths, not split mode")
        if self.share_scale_recurrent:
            first_pan = pan_channels_per_level[0]
            first_ms = ms_channels_per_level[0]
            if any(c != first_pan for c in pan_channels_per_level) or any(c != first_ms for c in ms_channels_per_level):
                raise ValueError("share_scale_recurrent requires equal PAN/MS feature channels at every wavelet level")

        if use_z_residual_head:
            self.z_res_heads = nn.ModuleList([
                ZResidualHead(hidden_dim, latent_dim, ms_channels_per_level[i])
                for i in range(levels)
            ])

        if use_local_freq_mixer:
            self.local_mixer_lh = nn.ModuleList([
                LocalFrequencyMixer(ms_channels_per_level[i], lfm_kernel_size, lfm_hidden_scale)
                for i in range(levels)
            ])
            self.local_mixer_hl = nn.ModuleList([
                LocalFrequencyMixer(ms_channels_per_level[i], lfm_kernel_size, lfm_hidden_scale)
                for i in range(levels)
            ])
            self.local_mixer_hh = nn.ModuleList([
                LocalFrequencyMixer(ms_channels_per_level[i], lfm_kernel_size, lfm_hidden_scale)
                for i in range(levels)
            ])

        if use_windowed_freq_mixer:
            self.window_mixer_lh = nn.ModuleList([
                WindowedFrequencyMixer(ms_channels_per_level[i], wfm_window_size, wfm_hidden_scale)
                for i in range(levels)
            ])
            self.window_mixer_hl = nn.ModuleList([
                WindowedFrequencyMixer(ms_channels_per_level[i], wfm_window_size, wfm_hidden_scale)
                for i in range(levels)
            ])

        if use_mamba_freq_mixer:
            self.mamba_mixer_lh = nn.ModuleList([
                MambaFrequencyMixer(
                    ms_channels_per_level[i], mamba_window_size, mamba_hidden_scale,
                    mamba_d_state, mamba_d_conv, mamba_expand
                )
                for i in range(levels)
            ])
            self.mamba_mixer_hl = nn.ModuleList([
                MambaFrequencyMixer(
                    ms_channels_per_level[i], mamba_window_size, mamba_hidden_scale,
                    mamba_d_state, mamba_d_conv, mamba_expand
                )
                for i in range(levels)
            ])
            self.mamba_mixer_hh = nn.ModuleList([
                MambaFrequencyMixer(
                    ms_channels_per_level[i], mamba_window_size, mamba_hidden_scale,
                    mamba_d_state, mamba_d_conv, mamba_expand
                )
                for i in range(levels)
            ])
            self.window_mixer_hh = nn.ModuleList([
                WindowedFrequencyMixer(ms_channels_per_level[i], wfm_window_size, wfm_hidden_scale)
                for i in range(levels)
            ])

        if self.freq_state_mode == "split":
            self.split_fusion_blocks = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            band: CrossScaleFusionHz(
                                pan_channels=pan_channels_per_level[i] // 4,
                                ms_channels=ms_channels_per_level[i],
                                hidden_dim=hidden_dim,
                                latent_dim=latent_dim,
                                deterministic_only=deterministic_only,
                                use_conv_gru=use_conv_gru,
                                state_conv_type=state_conv_type,
                                state_kernel_size=state_kernel_size,
                                use_state_spatial_mixer=use_state_spatial_mixer,
                                use_level_ll_corr=use_level_ll_corr if band == "ll" else False,
                                z_eval_mode=z_eval_mode,
                                z_update_order=z_update_order,
                            )
                            for band in ("ll", "lh", "hl", "hh")
                        }
                    )
                    for i in range(levels)
                ]
            )
        else:
            def _make_fusion_block(i):
                return CrossScaleFusionHz(
                    pan_channels=pan_channels_per_level[i],
                    ms_channels=ms_channels_per_level[i],
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    deterministic_only=deterministic_only,
                    use_conv_gru=use_conv_gru,
                    state_conv_type=state_conv_type,
                    state_kernel_size=state_kernel_size,
                    use_freq_gated_state=self.freq_state_mode == "simple",
                    gate_mod_channels=(pan_channels_per_level[i] // 4) * 3 + ms_channels_per_level[i] * 3,
                    use_state_spatial_mixer=use_state_spatial_mixer,
                    use_level_ll_corr=use_level_ll_corr,
                    z_eval_mode=z_eval_mode,
                    z_update_order=z_update_order,
                )

            if self.share_scale_recurrent:
                # Phase-1 regularization: the same state cell is reused from
                # coarse to fine, so scale identity is carried by inputs rather
                # than by three independent parameter sets.
                self.shared_fusion_block = _make_fusion_block(0)
            else:
                self.fusion_blocks = nn.ModuleList([_make_fusion_block(i) for i in range(levels)])

        def _make_state_up(channels):
            return nn.Sequential(
                nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=True),
                nn.PReLU(channels),
            )

        if self.share_scale_recurrent:
            self.shared_state_up_h = _make_state_up(hidden_dim)
            self.shared_state_up_z = _make_state_up(latent_dim)
        else:
            self.state_up_h = nn.ModuleList([_make_state_up(hidden_dim) for _ in range(levels - 1)])
            self.state_up_z = nn.ModuleList([_make_state_up(latent_dim) for _ in range(levels - 1)])

        self.separate_subband_gates = separate_subband_gates

        self.pan_high_to_ms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "lh": nn.Conv2d(pan_channels_per_level[i] // 4, ms_channels_per_level[i], 1, bias=True),
                        "hl": nn.Conv2d(pan_channels_per_level[i] // 4, ms_channels_per_level[i], 1, bias=True),
                        "hh": nn.Conv2d(pan_channels_per_level[i] // 4, ms_channels_per_level[i], 1, bias=True),
                    }
                )
                for i in range(levels)
            ]
        )

        # Project z_state (latent_dim) to MS channel space so it can inform the high-frequency gates.
        self.z_to_gate = nn.ModuleList(
            [
                nn.Conv2d(latent_dim, ms_channels_per_level[i], 1, bias=True)
                for i in range(levels)
            ]
        )

        # Gate input = [fused_ll, ll_ms, pan_lh, pan_hl, pan_hh, z_gate] -> 6 * ms_channels
        gate_in_channels = [ms_channels_per_level[i] * 6 for i in range(levels)]
        if separate_subband_gates:
            self.high_gate_lh = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(gate_in_channels[i], ms_channels_per_level[i], 1, bias=True), nn.Sigmoid())
                 for i in range(levels)]
            )
            self.high_gate_hl = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(gate_in_channels[i], ms_channels_per_level[i], 1, bias=True), nn.Sigmoid())
                 for i in range(levels)]
            )
            self.high_gate_hh = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(gate_in_channels[i], ms_channels_per_level[i], 1, bias=True), nn.Sigmoid())
                 for i in range(levels)]
            )
        else:
            self.high_gate = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(gate_in_channels[i], ms_channels_per_level[i], 1, bias=True), nn.Sigmoid())
                 for i in range(levels)]
            )

        # Learnable fusion: replace additive injection with small ConvNets.
        # Each ConvFusion takes [MS_hf, PAN_hf, alpha * PAN_hf] and produces fused output.
        if learnable_fusion:
            def _make_conv_fusion(in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=True),
                    nn.PReLU(out_ch),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
                    nn.PReLU(out_ch),
                    nn.Conv2d(out_ch, out_ch, 1, bias=True),
                )
            # Input: MS_hf (ms_c) + PAN_hf (ms_c) + alpha*PAN_hf (ms_c) = 3*ms_c
            fusion_in_channels = [ms_channels_per_level[i] * 3 for i in range(levels)]
            self.conv_fusion_lh = nn.ModuleList(
                [_make_conv_fusion(fusion_in_channels[i], ms_channels_per_level[i]) for i in range(levels)]
            )
            self.conv_fusion_hl = nn.ModuleList(
                [_make_conv_fusion(fusion_in_channels[i], ms_channels_per_level[i]) for i in range(levels)]
            )
            self.conv_fusion_hh = nn.ModuleList(
                [_make_conv_fusion(fusion_in_channels[i], ms_channels_per_level[i]) for i in range(levels)]
            )
            if residual_learnable_fusion:
                self.conv_fusion_beta_lh = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(levels)])
                self.conv_fusion_beta_hl = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(levels)])
                self.conv_fusion_beta_hh = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(levels)])

        self.wavelet_reconstructor = WaveletReconstructor(levels=levels)
        self.refine = nn.Sequential(resblock(ms_channels_per_level[0]), resblock(ms_channels_per_level[0]))

    def _fusion_block(self, level):
        if self.share_scale_recurrent:
            return self.shared_fusion_block
        return self.fusion_blocks[level]

    def _upsample_h(self, level, h_state):
        if self.share_scale_recurrent:
            return self.shared_state_up_h(h_state)
        return self.state_up_h[level](h_state)

    def _upsample_z(self, level, z_state):
        if self.share_scale_recurrent:
            return self.shared_state_up_z(z_state)
        return self.state_up_z[level](z_state)

    def _hf_alpha(self, alpha):
        if self.signed_hf_gate:
            # Keep the pretrained positive gate and add a small signed correction.
            # This can suppress or amplify PAN details without destroying old checkpoints at step 0.
            return alpha + self.hf_gate_scale * (2.0 * alpha - 1.0)
        return self.hf_gate_scale * alpha

    def _fuse_high(self, ms_hf, pan_hf, alpha, conv_fusion=None, beta=None):
        if self.learnable_fusion:
            delta = conv_fusion(torch.cat([ms_hf, pan_hf, alpha * pan_hf], dim=1))
            if self.residual_learnable_fusion:
                return ms_hf + alpha * pan_hf + beta * delta
            return delta
        return ms_hf + alpha * pan_hf

    def set_z_eval_mode(self, mode):
        if mode not in {"prior", "posterior", "zero"}:
            raise ValueError(f"Unsupported z_eval_mode: {mode}")
        self.z_eval_mode = mode
        if self.freq_state_mode == "split":
            for level_blocks in self.split_fusion_blocks:
                for block in level_blocks.values():
                    block.set_z_eval_mode(mode)
        elif self.share_scale_recurrent:
            self.shared_fusion_block.set_z_eval_mode(mode)
        else:
            for block in self.fusion_blocks:
                block.set_z_eval_mode(mode)

    def set_z_diagnostics(self, enabled=True):
        self.collect_z_diagnostics = bool(enabled)

    def get_z_diagnostics(self):
        return list(self.last_z_diagnostics)

    def _forward_split(self, pan_pyramid, ms_pyramid, training=True):
        bands = ("ll", "lh", "hl", "hh")
        h_states = {band: None for band in bands}
        z_states = {band: None for band in bands}
        fused_coeffs = []
        kl_terms = []
        self.last_z_diagnostics = []

        for level in range(self.levels - 1, -1, -1):
            ll_pan, lh_pan, hl_pan, hh_pan = pan_pyramid[level]
            ll_ms, lh_ms, hl_ms, hh_ms = ms_pyramid[level]
            pan_bands = {"ll": ll_pan, "lh": lh_pan, "hl": hl_pan, "hh": hh_pan}
            ms_bands = {"ll": ll_ms, "lh": lh_ms, "hl": hl_ms, "hh": hh_ms}
            fused_bands = {}
            b, _, h, w = ll_pan.shape
            force_zero_z = level in self.z_zero_levels

            for band in bands:
                h_prev = h_states[band]
                z_prev = z_states[band]
                if h_prev is None:
                    h_prev = torch.zeros(b, self.hidden_dim, h, w, device=ll_pan.device, dtype=ll_pan.dtype)
                    z_prev = torch.zeros(b, self.latent_dim, h, w, device=ll_pan.device, dtype=ll_pan.dtype)
                else:
                    h_prev = self._upsample_h(level, h_prev)
                    z_prev = self._upsample_z(level, z_prev)
                    if h_prev.shape[2:] != (h, w):
                        h_prev = F.interpolate(h_prev, size=(h, w), mode="bilinear", align_corners=False)
                    if z_prev.shape[2:] != (h, w):
                        z_prev = F.interpolate(z_prev, size=(h, w), mode="bilinear", align_corners=False)

                fused, h_new, z_new, kl_mean, _ = self.split_fusion_blocks[level][band](
                    pan_bands[band],
                    ms_bands[band],
                    h_prev,
                    z_prev,
                    training=training,
                    force_zero_z=force_zero_z,
                    return_z0=False,
                )
                fused_bands[band] = fused
                h_states[band] = h_new
                z_states[band] = z_new
                kl_terms.append(kl_mean)

            fused_coeffs.append((fused_bands["ll"], fused_bands["lh"], fused_bands["hl"], fused_bands["hh"]))

        fused_coeffs = list(reversed(fused_coeffs))
        recon = self.wavelet_reconstructor(fused_coeffs)
        recon = self.refine(recon)
        kl_loss = torch.stack(kl_terms).mean() if kl_terms else recon.new_tensor(0.0)
        return recon, kl_loss, [] if self.use_z_residual_head else None

    def forward(self, pan_pyramid, ms_pyramid, training=True):
        if self.freq_state_mode == "split":
            return self._forward_split(pan_pyramid, ms_pyramid, training=training)

        h_state = None
        z_state = None
        fused_coeffs = []
        kl_terms = []
        z_residuals = [] if self.use_z_residual_head else None
        self.last_z_diagnostics = []

        for level in range(self.levels - 1, -1, -1):
            ll_pan, lh_pan, hl_pan, hh_pan = pan_pyramid[level]
            ll_ms, lh_ms, hl_ms, hh_ms = ms_pyramid[level]
            b, _, h, w = ll_pan.shape

            pan_feat = torch.cat([ll_pan, lh_pan, hl_pan, hh_pan], dim=1)
            ms_feat = ll_ms
            gate_mod_feat = (
                torch.cat([lh_pan, hl_pan, hh_pan, lh_ms, hl_ms, hh_ms], dim=1)
                if self.freq_state_mode == "simple"
                else None
            )

            if h_state is None:
                h_state = torch.zeros(b, self.hidden_dim, h, w, device=ll_pan.device, dtype=ll_pan.dtype)
                z_state = torch.zeros(b, self.latent_dim, h, w, device=ll_pan.device, dtype=ll_pan.dtype)
            else:
                h_state = self._upsample_h(level, h_state)
                z_state = self._upsample_z(level, z_state)
                if h_state.shape[2:] != (h, w):
                    h_state = F.interpolate(h_state, size=(h, w), mode="bilinear", align_corners=False)
                if z_state.shape[2:] != (h, w):
                    z_state = F.interpolate(z_state, size=(h, w), mode="bilinear", align_corners=False)

            force_zero_z = level in self.z_zero_levels
            fused_ll, h_state, z_state, kl_mean, fused_ll_z0 = self._fusion_block(level)(
                pan_feat,
                ms_feat,
                h_state,
                z_state,
                training=training,
                force_zero_z=force_zero_z,
                return_z0=self.collect_z_diagnostics,
                gate_mod_feat=gate_mod_feat,
            )
            kl_terms.append(kl_mean)

            pan_lh = self.pan_high_to_ms[level]["lh"](lh_pan)
            pan_hl = self.pan_high_to_ms[level]["hl"](hl_pan)
            pan_hh = self.pan_high_to_ms[level]["hh"](hh_pan)

            z_gate = self.z_to_gate[level](z_state)
            gate_in = torch.cat([fused_ll, ll_ms, pan_lh, pan_hl, pan_hh, z_gate], dim=1)
            z_gate_zero = None
            gate_in_z0 = None
            if self.collect_z_diagnostics:
                z_gate_zero = self.z_to_gate[level](torch.zeros_like(z_state))
                fused_ll_base = fused_ll_z0 if fused_ll_z0 is not None else fused_ll
                gate_in_z0 = torch.cat([fused_ll_base, ll_ms, pan_lh, pan_hl, pan_hh, z_gate_zero], dim=1)

            if self.separate_subband_gates:
                alpha_lh = self._hf_alpha(self.high_gate_lh[level](gate_in))
                alpha_hl = self._hf_alpha(self.high_gate_hl[level](gate_in))
                alpha_hh = self._hf_alpha(self.high_gate_hh[level](gate_in))
                if self.collect_z_diagnostics:
                    alpha_lh_z0 = self._hf_alpha(self.high_gate_lh[level](gate_in_z0))
                    alpha_hl_z0 = self._hf_alpha(self.high_gate_hl[level](gate_in_z0))
                    alpha_hh_z0 = self._hf_alpha(self.high_gate_hh[level](gate_in_z0))
                beta_lh = self.conv_fusion_beta_lh[level] if self.learnable_fusion and self.residual_learnable_fusion else None
                beta_hl = self.conv_fusion_beta_hl[level] if self.learnable_fusion and self.residual_learnable_fusion else None
                beta_hh = self.conv_fusion_beta_hh[level] if self.learnable_fusion and self.residual_learnable_fusion else None
                fused_lh = self._fuse_high(
                    lh_ms, pan_lh, alpha_lh, self.conv_fusion_lh[level] if self.learnable_fusion else None, beta_lh
                )
                fused_hl = self._fuse_high(
                    hl_ms, pan_hl, alpha_hl, self.conv_fusion_hl[level] if self.learnable_fusion else None, beta_hl
                )
                fused_hh = self._fuse_high(
                    hh_ms, pan_hh, alpha_hh, self.conv_fusion_hh[level] if self.learnable_fusion else None, beta_hh
                )
            else:
                alpha = self._hf_alpha(self.high_gate[level](gate_in))
                if self.collect_z_diagnostics:
                    alpha_z0 = self._hf_alpha(self.high_gate[level](gate_in_z0))
                    alpha_lh_z0 = alpha_z0
                    alpha_hl_z0 = alpha_z0
                    alpha_hh_z0 = alpha_z0
                    alpha_lh = alpha
                    alpha_hl = alpha
                    alpha_hh = alpha
                beta_lh = self.conv_fusion_beta_lh[level] if self.learnable_fusion and self.residual_learnable_fusion else None
                beta_hl = self.conv_fusion_beta_hl[level] if self.learnable_fusion and self.residual_learnable_fusion else None
                beta_hh = self.conv_fusion_beta_hh[level] if self.learnable_fusion and self.residual_learnable_fusion else None
                fused_lh = self._fuse_high(
                    lh_ms, pan_lh, alpha, self.conv_fusion_lh[level] if self.learnable_fusion else None, beta_lh
                )
                fused_hl = self._fuse_high(
                    hl_ms, pan_hl, alpha, self.conv_fusion_hl[level] if self.learnable_fusion else None, beta_hl
                )
                fused_hh = self._fuse_high(
                    hh_ms, pan_hh, alpha, self.conv_fusion_hh[level] if self.learnable_fusion else None, beta_hh
                )

            if self.use_local_freq_mixer:
                fused_lh = fused_lh + self.local_mixer_lh[level](lh_ms, pan_lh, alpha_lh, fused_ll, ll_ms)
                fused_hl = fused_hl + self.local_mixer_hl[level](hl_ms, pan_hl, alpha_hl, fused_ll, ll_ms)
                fused_hh = fused_hh + self.local_mixer_hh[level](hh_ms, pan_hh, alpha_hh, fused_ll, ll_ms)
            if self.use_windowed_freq_mixer:
                fused_lh = fused_lh + self.window_mixer_lh[level](lh_ms, pan_lh, alpha_lh, fused_ll, ll_ms)
                fused_hl = fused_hl + self.window_mixer_hl[level](hl_ms, pan_hl, alpha_hl, fused_ll, ll_ms)
                fused_hh = fused_hh + self.window_mixer_hh[level](hh_ms, pan_hh, alpha_hh, fused_ll, ll_ms)
            if self.use_mamba_freq_mixer:
                fused_lh = fused_lh + self.mamba_mixer_lh[level](lh_ms, pan_lh, alpha_lh, fused_ll, ll_ms)
                fused_hl = fused_hl + self.mamba_mixer_hl[level](hl_ms, pan_hl, alpha_hl, fused_ll, ll_ms)
                fused_hh = fused_hh + self.mamba_mixer_hh[level](hh_ms, pan_hh, alpha_hh, fused_ll, ll_ms)

            # Z residual auxiliary head: apply zero-init residual correction
            z_res_pred = None
            if self.use_z_residual_head:
                r_ll, r_lh, r_hl, r_hh = self.z_res_heads[level](h_state, z_state)
                z_res_pred = (r_ll, r_lh, r_hl, r_hh)
                z_residuals.append(z_res_pred)
                fused_ll = fused_ll + self.z_res_heads[level].beta_ll * r_ll
                fused_lh = fused_lh + self.z_res_heads[level].beta_hf * r_lh
                fused_hl = fused_hl + self.z_res_heads[level].beta_hf * r_hl
                fused_hh = fused_hh + self.z_res_heads[level].beta_hf * r_hh

            fused_coeffs.append((fused_ll, fused_lh, fused_hl, fused_hh))

            if self.collect_z_diagnostics:
                ll_delta = (
                    (fused_ll - fused_ll_z0).abs().mean()
                    if fused_ll_z0 is not None
                    else fused_ll.new_tensor(0.0)
                )
                self.last_z_diagnostics.append(
                    {
                        "level": int(level),
                        "force_zero_z": bool(force_zero_z),
                        "z_abs_mean": float(z_state.detach().abs().mean().cpu()),
                        "z_std": float(z_state.detach().float().std(unbiased=False).cpu()),
                        "z_gate_abs_mean": float(z_gate.detach().abs().mean().cpu()),
                        "ll_delta_z0": float(ll_delta.detach().cpu()),
                        "alpha_lh_mean": float(alpha_lh.detach().mean().cpu()),
                        "alpha_hl_mean": float(alpha_hl.detach().mean().cpu()),
                        "alpha_hh_mean": float(alpha_hh.detach().mean().cpu()),
                        "alpha_lh_delta_z0": float((alpha_lh - alpha_lh_z0).detach().abs().mean().cpu()),
                        "alpha_hl_delta_z0": float((alpha_hl - alpha_hl_z0).detach().abs().mean().cpu()),
                        "alpha_hh_delta_z0": float((alpha_hh - alpha_hh_z0).detach().abs().mean().cpu()),
                        "kl_mean": float(kl_mean.detach().cpu()),
                    }
                )

        fused_coeffs = list(reversed(fused_coeffs))
        recon = self.wavelet_reconstructor(fused_coeffs)
        recon = self.refine(recon)

        kl_loss = torch.stack(kl_terms).mean() if kl_terms else recon.new_tensor(0.0)
        return recon, kl_loss, z_residuals


class RSSMHWViTHZ(nn.Module):
    def __init__(
        self,
        L_up_channel,
        pan_channel,
        pan_target_channel,
        ms_target_channel,
        hidden_dim=96,
        latent_dim=32,
        deterministic_only=False,
        separate_subband_gates=True,
        use_conv_gru=False,
        state_conv_type="plain",
        state_kernel_size=3,
        freq_state_mode="mixed",
        learnable_fusion=False,
        image_space_wavelet=False,
        use_lowfreq_corr=False,
        signed_hf_gate=False,
        hf_gate_scale=1.0,
        residual_learnable_fusion=False,
        use_sdem_lite=False,
        use_state_spatial_mixer=False,
        use_level_ll_corr=False,
        use_band_corr=False,
        band_corr_kernel_size=5,
        band_corr_hidden=32,
        z_eval_mode="prior",
        z_update_order="legacy",
        z_zero_levels=None,
        levels=3,
        use_z_residual_head=False,
        use_local_freq_mixer=False,
        lfm_kernel_size=3,
        lfm_hidden_scale=1.0,
        use_linear_freq_attention=False,
        linear_attn_heads=4,
        use_windowed_freq_mixer=False,
        wfm_window_size=8,
        wfm_hidden_scale=1.0,
        use_mamba_freq_mixer=False,
        mamba_window_size=8,
        mamba_hidden_scale=1.0,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        use_channel_dwt_adapter=False,
        channel_dwt_hidden=32,
        use_wfanet_two_stage=False,
        share_scale_recurrent=False,
    ):
        super().__init__()
        self.deterministic_only = deterministic_only
        self.image_space_wavelet = image_space_wavelet
        self.levels = levels
        self.use_sdem_lite = use_sdem_lite
        self.z_eval_mode = z_eval_mode
        self.z_update_order = z_update_order
        self.use_channel_dwt_adapter = use_channel_dwt_adapter
        self.use_wfanet_two_stage = use_wfanet_two_stage
        self.state_conv_type = state_conv_type
        self.state_kernel_size = state_kernel_size
        self.freq_state_mode = str(freq_state_mode)
        self.share_scale_recurrent = bool(share_scale_recurrent)

        if self.use_wfanet_two_stage and self.image_space_wavelet:
            raise ValueError("--use-wfanet-two-stage is a feature-space design and is incompatible with --image-space-wavelet")
        if self.use_wfanet_two_stage and self.freq_state_mode == "split":
            raise ValueError("--freq-state-mode split is implemented for the recursive wavelet path, not --use-wfanet-two-stage")

        self.pan_raise = raise_channel(in_channel=pan_channel, target_channel=pan_target_channel)
        self.ms_upsample = nn.Sequential(
            nn.Conv2d(L_up_channel, L_up_channel * 16, 3, 1, 1, bias=True),
            nn.PixelShuffle(4),
        )
        self.ms_act = nn.PReLU(num_parameters=L_up_channel, init=0.01)
        self.channel_dwt_adapter = (
            ChannelHaarSpectralAdapter(L_up_channel, hidden_channels=channel_dwt_hidden)
            if use_channel_dwt_adapter else None
        )
        self.ms_raise = raise_channel(in_channel=L_up_channel, target_channel=ms_target_channel)
        self.ms_lr_raise = (
            raise_channel(in_channel=L_up_channel, target_channel=ms_target_channel)
            if self.use_wfanet_two_stage else None
        )

        self.wavelet = WaveletPyramid(levels=levels)

        if image_space_wavelet:
            # Wavelet on raw images: PAN(1-ch) and MS_up(L_up_channel-ch).
            # Each PAN subband (1-ch) is raised to pan_target_channel via a shared raiser.
            self.pan_subband_raise = raise_channel(in_channel=pan_channel, target_channel=pan_target_channel)
            # Each MS subband (L_up_channel-ch) is raised to ms_target_channel via a shared raiser.
            self.ms_subband_raise = raise_channel(in_channel=L_up_channel, target_channel=ms_target_channel)

        pan_channels_per_level = [pan_target_channel * 4] * levels
        ms_channels_per_level = [ms_target_channel] * levels

        if self.use_wfanet_two_stage:
            self.rssm_fusion = WFANetTwoStageRSSMFusion(
                pan_channels=pan_target_channel,
                ms_channels=ms_target_channel,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                deterministic_only=deterministic_only,
                separate_subband_gates=separate_subband_gates,
                use_conv_gru=use_conv_gru,
                state_conv_type=state_conv_type,
                state_kernel_size=state_kernel_size,
                use_freq_gated_state=self.freq_state_mode == "simple",
                signed_hf_gate=signed_hf_gate,
                hf_gate_scale=hf_gate_scale,
                use_state_spatial_mixer=use_state_spatial_mixer,
                use_level_ll_corr=use_level_ll_corr,
                z_eval_mode=z_eval_mode,
                z_update_order=z_update_order,
                use_local_freq_mixer=use_local_freq_mixer,
                lfm_kernel_size=lfm_kernel_size,
                lfm_hidden_scale=lfm_hidden_scale,
                use_linear_freq_attention=use_linear_freq_attention,
                linear_attn_heads=linear_attn_heads,
                share_scale_recurrent=share_scale_recurrent,
            )
        else:
            self.rssm_fusion = RSSMWaveletFusionHz(
                pan_channels_per_level=pan_channels_per_level,
                ms_channels_per_level=ms_channels_per_level,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                levels=levels,
                deterministic_only=deterministic_only,
                separate_subband_gates=separate_subband_gates,
                use_conv_gru=use_conv_gru,
                state_conv_type=state_conv_type,
                state_kernel_size=state_kernel_size,
                freq_state_mode=self.freq_state_mode,
                learnable_fusion=learnable_fusion,
                signed_hf_gate=signed_hf_gate,
                hf_gate_scale=hf_gate_scale,
                residual_learnable_fusion=residual_learnable_fusion,
                use_state_spatial_mixer=use_state_spatial_mixer,
                use_level_ll_corr=use_level_ll_corr,
                z_eval_mode=z_eval_mode,
                z_update_order=z_update_order,
                z_zero_levels=z_zero_levels,
                use_z_residual_head=use_z_residual_head,
                use_local_freq_mixer=use_local_freq_mixer,
                lfm_kernel_size=lfm_kernel_size,
                lfm_hidden_scale=lfm_hidden_scale,
                use_windowed_freq_mixer=use_windowed_freq_mixer,
                wfm_window_size=wfm_window_size,
                wfm_hidden_scale=wfm_hidden_scale,
                use_mamba_freq_mixer=use_mamba_freq_mixer,
                mamba_window_size=mamba_window_size,
                mamba_hidden_scale=mamba_hidden_scale,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                share_scale_recurrent=share_scale_recurrent,
            )

        self.reduce = reduce_channel(ms_target_channel=ms_target_channel, L_up_channel=L_up_channel)
        self.lowfreq_corr = LowFreqCorrection(L_up_channel) if use_lowfreq_corr else None
        self.band_corr = BandAwareCorrection(
            L_up_channel,
            hidden_channels=band_corr_hidden,
            kernel_size=band_corr_kernel_size,
        ) if use_band_corr else None
        self.sdem_lite = SDEMLite(ms_target_channel) if use_sdem_lite else None
        self.out_act = nn.PReLU(num_parameters=L_up_channel, init=0.01)
        self.fused_weight = nn.Parameter(torch.ones(1))  # default 1.0 = original behavior

    def set_z_eval_mode(self, mode):
        self.z_eval_mode = mode
        self.rssm_fusion.set_z_eval_mode(mode)

    def set_z_diagnostics(self, enabled=True):
        self.rssm_fusion.set_z_diagnostics(enabled)

    def get_z_diagnostics(self):
        return self.rssm_fusion.get_z_diagnostics()

    def forward(self, pan, ms, lms):
        ms_up = self.ms_upsample(ms)
        ms_up = self.ms_act(ms_up + lms)
        if self.channel_dwt_adapter is not None:
            ms_up = self.channel_dwt_adapter(ms_up, lms, pan)

        if self.use_wfanet_two_stage:
            pan_feat = self.pan_raise(pan)
            ms_feat = self.ms_raise(ms_up)
            ms_lr_feat = self.ms_lr_raise(ms)
            pan_feat_for_sdem = pan_feat
            fused, kl_loss, z_residuals = self.rssm_fusion(
                pan_feat,
                ms_feat,
                ms_lr_feat,
                training=self.training,
            )
        elif self.image_space_wavelet:
            pan_feat_for_sdem = self.pan_raise(pan) if self.sdem_lite is not None else None
            # Wavelet decompose raw PAN and MS_up in image space.
            pan_pyr_raw = self.wavelet(pan)
            ms_pyr_raw = self.wavelet(ms_up)

            # Raise each subband from image-space channels to feature-space channels.
            pan_pyr = []
            ms_pyr = []
            for level in range(self.levels):
                ll_p, lh_p, hl_p, hh_p = pan_pyr_raw[level]   # 1-ch each
                ll_m, lh_m, hl_m, hh_m = ms_pyr_raw[level]   # L_up_channel-ch each

                ll_p_r = self.pan_subband_raise(ll_p)
                lh_p_r = self.pan_subband_raise(lh_p)
                hl_p_r = self.pan_subband_raise(hl_p)
                hh_p_r = self.pan_subband_raise(hh_p)

                ll_m_r = self.ms_subband_raise(ll_m)
                lh_m_r = self.ms_subband_raise(lh_m)
                hl_m_r = self.ms_subband_raise(hl_m)
                hh_m_r = self.ms_subband_raise(hh_m)

                pan_pyr.append((ll_p_r, lh_p_r, hl_p_r, hh_p_r))
                ms_pyr.append((ll_m_r, lh_m_r, hl_m_r, hh_m_r))
        else:
            pan_feat = self.pan_raise(pan)
            ms_feat = self.ms_raise(ms_up)
            pan_feat_for_sdem = pan_feat
            pan_pyr = self.wavelet(pan_feat)
            ms_pyr = self.wavelet(ms_feat)

        if not self.use_wfanet_two_stage:
            fused, kl_loss, z_residuals = self.rssm_fusion(pan_pyr, ms_pyr, training=self.training)
        if self.sdem_lite is not None:
            fused = fused + self.sdem_lite(pan_feat_for_sdem)
        fused = self.reduce(fused)
        delta_lf = self.lowfreq_corr(fused, ms_up, pan) if self.lowfreq_corr is not None else 0.0
        base = self.fused_weight * fused + ms_up + delta_lf
        delta_band = self.band_corr(base, fused, ms_up, pan) if self.band_corr is not None else 0.0
        out = self.out_act(base + delta_band)
        return out, kl_loss, z_residuals


# Public entry point. WSRNet is RSSMHWViTHZ configured with the shared
# scale-recurrent cell + deterministic hidden state (see train.build_model
# "wsrnet"). The alias keeps the published name while preserving checkpoint
# key compatibility.
WSRNet = RSSMHWViTHZ


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RSSMHWViTHZ(
        L_up_channel=8,
        pan_channel=1,
        pan_target_channel=32,
        ms_target_channel=32,
        hidden_dim=96,
        latent_dim=32,
    ).to(device)
    pan = torch.randn(2, 1, 64, 64, device=device)
    ms = torch.randn(2, 8, 16, 16, device=device)
    lms = torch.randn(2, 8, 64, 64, device=device)
    out, kl, _ = model(pan, ms, lms)
    print(out.shape, kl.item())
