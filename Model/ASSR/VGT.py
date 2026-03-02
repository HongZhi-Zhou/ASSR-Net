import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi


class WaveletTransform1D(nn.Module):
    def __init__(self, wavelet_scales=3):
        super().__init__()
        self.wavelet_scales = wavelet_scales
        self.filters = nn.ModuleList()
        for i in range(wavelet_scales):
            kernel_size = 3 + 2 * i
            conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                             padding=kernel_size // 2, bias=False)
            self._init_gaussian_filter(conv, kernel_size)
            self.filters.append(conv)

    def _init_gaussian_filter(self, conv, kernel_size):
        x = torch.arange(kernel_size).float() - kernel_size // 2
        weights = torch.exp(-x ** 2 / (2 * (kernel_size / 4) ** 2))
        weights = weights / weights.sum()
        with torch.no_grad():
            conv.weight.data = weights.view(1, 1, -1)

    def forward(self, x):
        B, C, L = x.shape
        if L < 4:
            coeffs = x.unsqueeze(2).repeat(1, 1, self.wavelet_scales, 1)
            return coeffs
        all_coeffs = []
        current_signal = x
        for scale in range(self.wavelet_scales):
            signal_reshaped = current_signal.reshape(B * C, 1, L)
            filtered = self.filters[scale](signal_reshaped)
            if scale > 0 and L > 4:
                downsampled = F.avg_pool1d(filtered, kernel_size=2, stride=2)
                upsampled = F.interpolate(downsampled, size=L,
                                          mode='linear', align_corners=False)
                coeff = upsampled
            else:
                coeff = filtered
            coeff = coeff.reshape(B, C, L)
            all_coeffs.append(coeff)
            if scale < self.wavelet_scales - 1 and L > 4:
                current_signal = F.avg_pool1d(current_signal, kernel_size=2, stride=2)
                L = L // 2
                if L < 4:
                    break
        target_length = all_coeffs[0].shape[-1]
        processed_coeffs = []
        for i, coeff in enumerate(all_coeffs):
            if coeff.shape[-1] != target_length:
                coeff_resized = F.interpolate(coeff, size=target_length,
                                              mode='linear', align_corners=False)
                processed_coeffs.append(coeff_resized)
            else:
                processed_coeffs.append(coeff)

        while len(processed_coeffs) < self.wavelet_scales:
            processed_coeffs.append(processed_coeffs[-1])
        coeff_stack = torch.stack(processed_coeffs[:self.wavelet_scales], dim=2)
        return coeff_stack


# 可变角度预测
class AnglePredictor(nn.Module):
    def __init__(self, in_channels, max_angles=8):
        super().__init__()
        self.max_angles = max_angles
        self.feature_extractor = nn.AdaptiveAvgPool2d((4, 4))
        self.angle_predictor = nn.Sequential(
            nn.Linear(in_channels * 16, 32),
            nn.ReLU(),
            nn.Linear(32, max_angles)
        )
        self.register_buffer('base_angles', torch.linspace(0, 180, max_angles))

    def forward(self, x):
        B, C, H, W = x.shape
        features = self.feature_extractor(x)
        features = features.view(B, -1)
        angle_offsets = self.angle_predictor(features) * 30
        angles = self.base_angles.unsqueeze(0) + angle_offsets
        angles = torch.clamp(angles, 0, 180)
        # sorted_angels = angles[:,:K]
        sorted_angles, _ = torch.sort(angles, dim=1)
        return sorted_angles, torch.ones(B, self.max_angles, device=x.device) * 0.5

# Radon
class RadonTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def _create_rotation_grids(self, H, W, angles, device):
        B, K = angles.shape

        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        y_coords, x_coords = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([x_coords, y_coords], dim=-1)  # [H, W, 2]

        coords_expanded = coords.unsqueeze(0).unsqueeze(0).expand(B, K, H, W, 2)

        theta_rad = angles * (torch.pi / 180)  # [B, K]
        cos_theta = torch.cos(theta_rad)  # [B, K]
        sin_theta = torch.sin(theta_rad)  # [B, K]

        rot_mat = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),  # [B, K, 2]
            torch.stack([sin_theta, cos_theta], dim=-1)  # [B, K, 2]
        ], dim=-2)  # [B, K, 2, 2]

        rotated_grids = torch.einsum('bkhwc,bkcd->bkhwd', coords_expanded, rot_mat)

        return rotated_grids.view(B * K, H, W, 2)

    def forward(self, x, angles):
        B, C, H, W = x.shape
        device = x.device
        if angles.numel() == 0:
            return torch.zeros(B, C, H, 1, device=device)
        grids = self._create_rotation_grids(H, W, angles, device)  # [B*K, H, W, 2]
        K = angles.shape[1]
        x_expanded = x.unsqueeze(1).expand(B, K, C, H, W).contiguous().view(B * K, C, H, W)
        try:
            rotated = F.grid_sample(
                x_expanded,
                grids,
                align_corners=True,
                mode='bilinear',
                padding_mode='zeros'
            )  # [B*K, C, H, W]

            projections = torch.sum(rotated, dim=3, keepdim=True)
            sinogram = projections.view(B, K, C, H, 1).permute(0, 2, 3, 4, 1).squeeze(3)
            return sinogram
        except Exception as e:
            print(f"Radon变换错误: {e}")
            return torch.zeros(B, C, H, K, device=device)


class UGT(nn.Module):
    def __init__(self, in_channels, max_angles=6, wavelet_scales=2, min_size=8):
        super().__init__()
        self.in_channels = in_channels
        self.wavelet_scales = wavelet_scales
        self.min_size = min_size
        self.angle_predictor = AnglePredictor(in_channels, max_angles)
        self.radon_transform = RadonTransform()
        self.wavelet_transform = WaveletTransform1D(wavelet_scales)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * wavelet_scales, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        try:
            angles, _ = self.angle_predictor(x)
            K = angles.shape[1] if angles.numel() > 0 else 0
            if K == 0:
                return x
            sinogram = self.radon_transform(x, angles)
            B, C, H_proj, K = sinogram.shape
            sinogram_reshaped = sinogram.permute(0, 3, 1, 2).contiguous().view(B * K, C, H_proj)
            wavelet_coeffs = self.wavelet_transform(sinogram_reshaped)
            wavelet_coeffs = wavelet_coeffs.view(B, K, C, self.wavelet_scales, H_proj)
            wavelet_coeffs = wavelet_coeffs.permute(0, 2, 4, 3, 1)  # [B, C, H_proj, scales, K]
            combined = wavelet_coeffs.contiguous().view(B, C * self.wavelet_scales, H_proj, K)
            if combined.shape[2] != H or combined.shape[3] != W:
                combined = F.interpolate(combined, size=(H, W),
                                         mode='bilinear', align_corners=True)
            if combined.size(1) != self.in_channels:
                output = self.fusion(combined)
            else:
                output = combined
            return output
        except Exception as e:
            print(f"脊波变换错误: {e}")
            return x


def test_UGT():
    print("test")
    test_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]
    for H, W in test_sizes:
        ridgelet = UGT(
            in_channels=32,
            max_angles=64,
            wavelet_scales=2,
            min_size=8
        )
        x = torch.randn(2, 32, H, W)
        output = ridgelet(x)
        print(f"成功: 输入 {x.shape} -> 输出 {output.shape}")


if __name__ == "__main__":
    # 运行测试
    test_UGT()