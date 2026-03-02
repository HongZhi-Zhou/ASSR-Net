import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi


class FastArousSubbandDecomposition(nn.Module):
    def __init__(self, in_channels, num_scales=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales
        self.gaussian_filters = nn.ModuleList()
        
        # 为每个尺度创建卷积层
        for i in range(num_scales):
            kernel_size = 2 * i + 5
            conv_x = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), 
                             padding=(0, kernel_size//2), bias=False, groups=in_channels)
            conv_y = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1),
                             padding=(kernel_size//2, 0), bias=False, groups=in_channels)
            
            self._init_separable_gaussian(conv_x, conv_y, kernel_size)
            self.gaussian_filters.append(nn.Sequential(conv_x, conv_y))
    
    def _init_separable_gaussian(self, conv_x, conv_y, kernel_size):
        """初始化可分离高斯滤波器"""
        # 1D高斯核
        x = torch.arange(kernel_size).float() - kernel_size // 2
        sigma = max(kernel_size / 6.0, 1.0)
        weights_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        weights_1d = weights_1d / weights_1d.sum()
        
        with torch.no_grad():
            # 扩展权重到所有通道
            weights_x = weights_1d.view(1, 1, 1, -1).repeat(self.in_channels, 1, 1, 1)
            weights_y = weights_1d.view(1, 1, -1, 1).repeat(self.in_channels, 1, 1, 1)
            
            conv_x.weight.data = weights_x
            conv_y.weight.data = weights_y
    
    def forward(self, x):
        B, C, H, W = x.shape
        subbands = []
        current = x
        
        for i in range(self.num_scales):
            # 直接处理所有通道
            filtered = self.gaussian_filters[i](current)
            subbands.append(filtered)
            
            # 为下一尺度准备（下采样）- 添加尺寸检查
            if i < self.num_scales - 1:
                # 检查当前尺寸是否足够大
                current_H, current_W = current.shape[-2:]
                if current_H > 2 and current_W > 2:
                    current = F.avg_pool2d(current, kernel_size=2, stride=2)
                else:
                    # 如果尺寸太小，停止下采样，用当前信号填充剩余尺度
                    for j in range(i + 1, self.num_scales):
                        subbands.append(current)
                    break
        
        # 确保子带数量正确
        while len(subbands) < self.num_scales:
            subbands.append(subbands[-1] if subbands else x)
        
        # 计算细节子带
        detail_subbands = []
        for i in range(self.num_scales - 1):
            current_band = subbands[i]
            next_band = subbands[i + 1]
            
            # 保持尺寸一致
            if next_band.shape[-2:] != current_band.shape[-2:]:
                next_band = F.interpolate(next_band, size=current_band.shape[-2:], 
                                        mode='bilinear', align_corners=True)
            
            detail = current_band - next_band
            detail_subbands.append(detail)
        
        coarse = subbands[-1]
        return detail_subbands, coarse


class FastWaveletTransform1D(nn.Module):
    """一维小波变换"""
    def __init__(self, wavelet_scales=3):
        super().__init__()
        self.wavelet_scales = wavelet_scales
        
        # 预计算所有尺度的滤波器
        self.filters = nn.ModuleList()
        for i in range(wavelet_scales):
            kernel_size = 3 + 2 * i
            # 使用正确的通道数设置
            conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                           padding=kernel_size//2, bias=False)
            self._init_gaussian_filter(conv, kernel_size)
            self.filters.append(conv)

    def _init_gaussian_filter(self, conv, kernel_size):
        """初始化高斯滤波器"""
        x = torch.arange(kernel_size).float() - kernel_size // 2
        weights = torch.exp(-x ** 2 / (2 * (kernel_size / 4) ** 2))
        weights = weights / weights.sum()
        with torch.no_grad():
            conv.weight.data = weights.view(1, 1, -1)

    def forward(self, x):
        """小波变换"""
        B, C, L = x.shape
        
        if L < 4:
            return x.unsqueeze(2).repeat(1, 1, self.wavelet_scales, 1)
        
        all_coeffs = []
        current = x
        
        for scale in range(self.wavelet_scales):
            # 重塑为 [B*C, 1, L] 
            current_reshaped = current.reshape(B * C, 1, L)
            filtered = self.filters[scale](current_reshaped)
            # 计算细节系数
            if scale > 0 and L > 4:
                downsampled = F.avg_pool1d(filtered, kernel_size=2, stride=2)
                upsampled = F.interpolate(downsampled, size=L, mode='linear', align_corners=False)
                coeff = filtered - upsampled
            else:
                coeff = filtered
                
            coeff = coeff.reshape(B, C, L)
            all_coeffs.append(coeff)
            
            # 下采样
            if scale < self.wavelet_scales - 1 and L > 4:
                current = F.avg_pool1d(current, kernel_size=2, stride=2)
                new_L = L // 2
                if new_L < 2:  # 如果长度太小，停止下采样
                    break
                L = new_L
        # 统一长度
        target_length = all_coeffs[0].shape[-1] if all_coeffs else x.shape[-1]
        processed_coeffs = []
        for coeff in all_coeffs:
            if coeff.shape[-1] != target_length:
                coeff = F.interpolate(coeff, size=target_length, mode='linear', align_corners=False)
            processed_coeffs.append(coeff)
        # 堆叠结果
        while len(processed_coeffs) < self.wavelet_scales:
            processed_coeffs.append(processed_coeffs[-1] if processed_coeffs else x)
            
        return torch.stack(processed_coeffs[:self.wavelet_scales], dim=2)


class FastRadonTransform(nn.Module):
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


class FastDirectionPredictor(nn.Module):
    def __init__(self, in_channels, max_directions=8):
        super().__init__()
        self.max_directions = max_directions
        
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(in_channels, max(8, in_channels//4), 3, padding=1),
            nn.ReLU(),
        )
        
        self.direction_predictor = nn.Sequential(
            nn.Linear(max(8, in_channels//4) * 64, 32),
            nn.ReLU(),
            nn.Linear(32, max_directions)
        )
        
        self.register_buffer('base_angles', torch.linspace(0, 180, max_directions))

    def forward(self, x):
        B, C, H, W = x.shape
        
        features = self.feature_extractor(x)
        features = features.view(B, -1)
        
        angle_offsets = torch.tanh(self.direction_predictor(features)) * 30
        angles = self.base_angles.unsqueeze(0) + angle_offsets
        angles = torch.clamp(angles, 0, 180)
        
        sorted_angles, _ = torch.sort(angles, dim=1)
        return sorted_angles, torch.ones(B, self.max_directions, device=x.device) * 0.5


class UGT(nn.Module):
    def __init__(self, in_channels, num_scales=3, max_directions=6, wavelet_scales=2, min_size=8):
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales
        self.max_directions = max_directions
        self.wavelet_scales = wavelet_scales
        self.min_size = min_size
        
        self.subband_decomp = FastArousSubbandDecomposition(in_channels, num_scales)
        self.direction_predictor = FastDirectionPredictor(in_channels, max_directions)
        self.radon_transform = FastRadonTransform()
        self.wavelet_transform = FastWaveletTransform1D(wavelet_scales)
        
        self.scale_fusions = nn.ModuleList()
        for i in range(num_scales - 1):
            self.scale_fusions.append(
                nn.Sequential(
                    nn.Conv2d(in_channels * wavelet_scales, in_channels, 1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_channels * num_scales, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def _process_single_scale(self, subband, angles, scale_idx):
        B, C, H, W = subband.shape
        if H < 4 or W < 4:
            return subband 
        sinogram = self.radon_transform(subband, angles)  # [B, C, H, K]
        K = sinogram.shape[-1]
        sinogram_reshaped = sinogram.permute(0, 3, 1, 2).contiguous()  # [B, K, C, H]
        sinogram_reshaped = sinogram_reshaped.view(B * K, C, H)  # [B*K, C, H]
        wavelet_coeffs = self.wavelet_transform(sinogram_reshaped)  # [B*K, C, scales, H]
        # 重构
        wavelet_coeffs = wavelet_coeffs.view(B, K, C, self.wavelet_scales, H)
        wavelet_coeffs = wavelet_coeffs.permute(0, 2, 4, 3, 1).contiguous()  # [B, C, H, scales, K]
        combined = wavelet_coeffs.reshape(B, C * self.wavelet_scales, H, K)
        # 尺寸调整
        if combined.shape[-2:] != (H, W):
            combined = F.interpolate(combined, size=(H, W), mode='bilinear', align_corners=True)
        # 通道融合
        if combined.size(1) != self.in_channels:
            combined = self.scale_fusions[scale_idx](combined)
        return combined

    def forward(self, x):
        B, C, H, W = x.shape
            # 子带分解
        detail_subbands, coarse = self.subband_decomp(x)
            # 方向预测
        angles, _ = self.direction_predictor(x)
            # 处理所有细节子带
        scale_outputs = []
        for scale_idx, subband in enumerate(detail_subbands):
            scale_output = self._process_single_scale(subband, angles, scale_idx)
            scale_outputs.append(scale_output)
            # 处理粗糙子带
        if coarse.shape[-2:] != (H, W):
            coarse = F.interpolate(coarse, size=(H, W), mode='bilinear', align_corners=True)
        scale_outputs.append(coarse)
            
        if len(scale_outputs) > 1:
            unified_outputs = []
            for output in scale_outputs:
                if output.shape[-2:] != (H, W):
                    output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)
                unified_outputs.append(output)
                
            combined = torch.cat(unified_outputs, dim=1)
                
            if combined.size(1) != self.in_channels:
                output = self.final_fusion(combined)
            else:
                output = combined
        else:
            output = scale_outputs[0]
        return output