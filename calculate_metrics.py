import os
import numpy as np
import torch
from hdf5storage import loadmat
from torch import nn
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        Itrue = im_true.clamp(0., 1.)*data_range
        Ifake = im_fake.clamp(0., 1.)*data_range
        err=Itrue-Ifake
        err=torch.pow(err,2)
        err = torch.mean(err,dim=0)
        err = torch.mean(err,dim=0)

        psnr = 10. * torch.log10((data_range ** 2) / err)
        psnr=torch.mean(psnr)
        return psnr


class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs.clamp(0., 1.)*255- label.clamp(0., 1.)*255
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse

class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()
        self.eps=2.2204e-16
    def forward(self,im1, im2):
        assert im1.shape == im2.shape
        H,W,C=im1.shape
        im1 = np.reshape(im1,( H*W,C))
        im2 = np.reshape(im2,(H*W,C))
        core=np.multiply(im1, im2)
        mole = np.sum(core, axis=1)
        im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        deno = np.multiply(im1_norm, im2_norm)
        sam = np.rad2deg(np.arccos(((mole+self.eps)/(deno+self.eps)).clip(-1,1)))
        return np.mean(sam)


def qnr_index(fused, msi, hsi, alpha=1, beta=1):
    """
    计算 QNR 指标
    :param fused: (H, W, B) 融合结果 (HRHSI)
    :param msi:   (H, W, C) 原始 HRMSI
    :param hsi:   (H, W, B) 原始 LRHSI 上采样到 HR 尺度
    :param alpha: 光谱权重
    :param beta:  空间权重
    :return: QNR 值, D_lambda, D_s
    """
    fused = np.clip(fused, 0, 1).astype(np.float64)
    msi   = np.clip(msi, 0, 1).astype(np.float64)
    hsi   = np.clip(hsi, 0, 1).astype(np.float64)

    H, W, B = fused.shape
    _, _, C = msi.shape

    # --- 保证尺寸一致 ---
    if hsi.shape[:2] != (H, W):
        hsi = resize(hsi, (H, W, B), order=1, preserve_range=True, anti_aliasing=True)
    if msi.shape[:2] != (H, W):
        msi = resize(msi, (H, W, C), order=1, preserve_range=True, anti_aliasing=True)

    # --- 光谱失真 Dλ ---
    D_lambda = 0
    for i in range(B):
        for j in range(i + 1, B):
            corr_f = np.corrcoef(fused[:, :, i].ravel(), fused[:, :, j].ravel())[0, 1]
            corr_h = np.corrcoef(hsi[:, :, i].ravel(), hsi[:, :, j].ravel())[0, 1]
            D_lambda += abs(corr_f - corr_h)
    D_lambda /= (B * (B - 1) / 2)

    # --- 空间失真 Ds ---
    D_s = 0
    for c in range(C):
        psnr_f = psnr(msi[:, :, c], fused[:, :, c], data_range=1.0)
        psnr_h = psnr(msi[:, :, c], hsi[:, :, c], data_range=1.0)
        D_s += abs(psnr_f - psnr_h) / (abs(psnr_h) + 1e-6)
    D_s /= C

    # --- QNR ---
    QNR = (1 - D_lambda) ** alpha * (1 - D_s) ** beta
    return QNR

if __name__ == '__main__':
    SAM=Loss_SAM()
    RMSE=Loss_RMSE()
    PSNR=Loss_PSNR()
    psnr_list=[]
    sam_list=[]
    sam=AverageMeter()
    rmse=AverageMeter()
    psnr=AverageMeter()
    path1 = r'C:\Users\keff\Desktop\Fusion_code\result\Cave\Rea\\'
    path2 = r'C:\Users\keff\Desktop\Fusion_code\result\Cave\Fak\\'
    imglist = os.listdir(path1)
    for i in range(0, len(imglist)):
        img1 = loadmat(path2 + imglist[i])
        img2 = loadmat(path1 + imglist[i])
        # print(img2)
        lable = img1["fak"]
        recon = img2["rea"]
        sam_temp=SAM(lable,recon)
        psnr_temp=PSNR(torch.Tensor(lable), torch.Tensor(recon))
        sam.update(sam_temp)
        rmse.update(RMSE(torch.Tensor(lable),torch.Tensor(recon)))
        psnr.update(psnr_temp)
        psnr_list.append(psnr_temp)
        sam_list.append(sam_temp)
    print(sam.avg)
    print(rmse.avg)
    print(psnr.avg.cpu().detach().numpy())
    print(psnr_list)
    print(sam_list)