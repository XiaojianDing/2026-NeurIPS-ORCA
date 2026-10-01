import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, device):
        super(Network, self).__init__()
        self.view = view
        self.device = device

        self.encoders = nn.ModuleList([Encoder(input_size[v], feature_dim).to(device) for v in range(view)])
        self.decoders = nn.ModuleList([Decoder(input_size[v], feature_dim).to(device) for v in range(view)])

        self.feature_fusion_module = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, high_feature_dim)
        )

        self.common_information_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim)
        )

        self.gamma = nn.Parameter(torch.tensor(0.1, device=device))

        self.self_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True).to(device)

        self.global_query = nn.Parameter(torch.randn(1, 1, feature_dim, device=device))
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True).to(device)

        self.info_calibration = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        ).to(device)

    def orthogonal_enhancement(self, zs):
        enhanced_zs = []
        V = len(zs)
        if V <= 1:
            return zs

        for v in range(V):
            z_core = zs[v]
            residual_sum = torch.zeros_like(z_core)

            for u in range(V):
                if u == v:
                    continue
                z_other = zs[u]

                dot_product = torch.sum(z_other * z_core, dim=1, keepdim=True)
                norm_core_sq = torch.sum(z_core * z_core, dim=1, keepdim=True) + 1e-8
                z_proj = (dot_product / norm_core_sq) * z_core

                z_residual = z_other - z_proj

                norm_residual = torch.norm(z_residual, dim=1, keepdim=True)
                norm_other = torch.norm(z_other, dim=1, keepdim=True) + 1e-8
                g = norm_residual / norm_other

                residual_sum += g * z_residual

            residual_avg = residual_sum / (V - 1)
            z_new = z_core + residual_avg
            enhanced_zs.append(z_new)

        return enhanced_zs

    def compute_information_gain(self, current_Z, previous_Z):

        P = F.softmax(current_Z, dim=-1)
        log_Q = F.log_softmax(previous_Z, dim=-1)

        kl_gain = F.kl_div(log_Q, P, reduction='batchmean')
        return kl_gain.item()

    def hierarchical_fusion(self, zs):
        B = zs[0].shape[0]
        V = len(zs)

        Z = torch.stack(zs, dim=1)
        original_Z = Z.clone()

        current_Z = Z
        max_iters = 10
        threshold = 1e-4

        intermediate_zs = []
        final_H = None

        for iteration in range(max_iters):

            attn_out, _ = self.self_attn(current_Z, current_Z, current_Z)
            low_level_context = original_Z.mean(dim=1, keepdim=True)
            fused_Z = attn_out + self.gamma * low_level_context


            Q = self.global_query.expand(B, -1, -1)  # (Batch, 1, Feature_dim)
            H_out, _ = self.cross_attn(Q, fused_Z, fused_Z)
            H_out = H_out.squeeze(1)  # (Batch, Feature_dim)


            info_gain = self.compute_information_gain(fused_Z, current_Z)


            for v in range(V):
                intermediate_zs.append(fused_Z[:, v, :])


            if info_gain <= threshold or iteration == max_iters - 1:

                final_H = H_out
                break
            else:
                current_Z = fused_Z + self.info_calibration(fused_Z)

        return final_H, intermediate_zs

    def feature_fusion(self, zs):

        H, intermediate_zs = self.hierarchical_fusion(zs)
        H_final = self.feature_fusion_module(H)
        return normalize(H_final, dim=1), intermediate_zs

    def compute_fusion_weight(self, z1, z2):
        sim1 = torch.norm(z1, dim=1, keepdim=True)
        sim2 = torch.norm(z2, dim=1, keepdim=True)
        weight1 = sim1 / (sim1 + sim2 + 1e-8)
        weight2 = sim2 / (sim1 + sim2 + 1e-8)
        return weight1, weight2

    def fuse_views(self, zs):
        new_zs = []
        for i in range(self.view):
            for j in range(i + 1, self.view):
                weight1, weight2 = self.compute_fusion_weight(zs[i], zs[j])
                new_z = weight1 * zs[i] + weight2 * zs[j]
                new_zs.append(new_z)
        return new_zs

    def forward(self, xs):
        rs, xrs, zs = [], [], []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            r = normalize(self.common_information_module(z), dim=1)
            rs.append(r)
            zs.append(z)
            xrs.append(xr)

        new_zs = self.fuse_views(zs)
        new_rs = [normalize(self.common_information_module(new_z), dim=1) for new_z in new_zs]

        enhanced_zs = self.orthogonal_enhancement(zs)

        H, intermediate_zs = self.feature_fusion(enhanced_zs)

        intermediate_rs = [normalize(self.common_information_module(iz), dim=1) for iz in intermediate_zs]

        return xrs, zs, rs, H, new_rs, intermediate_rs