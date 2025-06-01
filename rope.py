import torch
import math

dim = 32
max_seq = 1024


def ref_rotate(x, pos, base=10000.0):
    d = x.size(-1) // 2
    freqs = (torch.arange(0, d, device=x.device).float() * -math.log(base) / d).exp()
    theta = pos.unsqueeze(-1) * freqs  # broadcast (seq_len, 1) × (dim//2) → (seq_len, dim//2)
    c, s = torch.cos(theta), torch.sin(theta)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


def precompute_freqs_cis(
    dim: int,
    max_seq: int,
    device: torch.device = torch.device("cuda"),
    theta: float = 10000.0,
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # [seq, dim//2]
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


cached_freqs_cos, cached_freqs_sin = precompute_freqs_cis(dim, max_seq)


class RotaryEmbedding:
    @staticmethod
    def forward(x, position_ids):
        # x shape: (seq_len, dim)
        cos = cached_freqs_cos[position_ids]  # [seq_len, dim//2]
        sin = cached_freqs_sin[position_ids]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        x[..., ::2] = rot1
        x[..., 1::2] = rot2
        return x


T = 3
x = torch.randn(T, dim, device=cached_freqs_cos.device)
position_ids = torch.arange(T, device=x.device)

out = RotaryEmbedding.forward(x.clone(), position_ids)
ref_out = ref_rotate(x, position_ids)

torch.testing.assert_close(out, ref_out, atol=1e-6, rtol=1e-6)
print("Passed")
