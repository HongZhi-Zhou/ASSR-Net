# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:41:48 2020

@author: Dian
"""
import time

import numpy as np

import torch
from scipy.io import loadmat

import os

from torch import nn

from calculate_metrics import Loss_SAM, Loss_RMSE, Loss_PSNR, qnr_index
# from models.Fuformer import MainNet
# from models.HSRnet import RGBNet
# from models.SSRnet import SSRNET
# from models.fusion_model_v15 import Cross_Guide_Fusion
# from models.model_guide import VSR_CAS
# from models.model_guide import VSR_CAS
# from MIMOgf import *
from Model.ASSR.VGSRgf import *
# from Train_MIMO.Model import *
from utils import create_F, Gaussian_downsample, fspecial, AverageMeter
# from SSGTN.SSGTN_gf5 import Cross_Guide_Fusion
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import hdf5storage

# dataset = 'Harvard'
# path = 'D:\data\Harvard\harvard_test/'

# dataset = 'CAVE'
# path='D:\data\cave\cave_test/'


# imglist = os.listdir(path)

model_path = r'PKL/GaoFen/VGSR/5/gf5_pkl/100EPOCH.pkl'
# model_path = r'PKL/GaoFen/MIMO/3/gf5_pkl/2000EPOCH_PSNR_best.pkl'
# net2 = Cross_Guide_Fusion(150, 64, 64,1).cuda()
# net2=SSRNET('SSRNET',8,3,31).cuda()
# net2 = VSR_CAS(channel0=31, factor=8, P=torch.Tensor(R)).cuda()
# net2 = RGBNet(150,1,200).cuda()
# net2 = Net_s().cuda()
# net2 =MainNet().cuda()
net2 = VGSR_NET(150,4,64).cuda()
# net2 = Net_sgf(150,4).cuda()
checkpoint = torch.load(model_path)  # 加载断点
# net2.load_state_dict(checkpoint['model_state_dict'])  # 加载模型可学习参数
net2.load_state_dict(checkpoint)
save_path = r"./result/DHIF-Net/Gaofen/"
def mkdir(path):

    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("测试保存文件夹为：{}".format(path))
    else:
        print('已存在{}'.format(path))
mkdir(save_path)
RMSE = []

loss_func = nn.L1Loss(reduction='mean').cuda()


def reconstruction_fg5(net2, R, HSI_LR, MSI_HR,HSI_HR, downsample_factor,training_size, stride,val_loss):
    index_matrix = torch.zeros((R.shape[1], MSI_HR.shape[2], MSI_HR.shape[3])).cuda()
    abundance_t = torch.zeros((R.shape[1], MSI_HR.shape[2], MSI_HR.shape[3])).cuda()
    a = []
    for j in range(0, MSI_HR.shape[2] - training_size + 1, stride):
        a.append(j)
    a.append(MSI_HR.shape[2] - training_size)
    b = []
    for j in range(0, MSI_HR.shape[3] - training_size + 1, stride):
        b.append(j)
    b.append(MSI_HR.shape[3] - training_size)
    for j in a:
        for k in b:
            temp_hrms = MSI_HR[:, :, j:j + training_size, k:k + training_size]
            temp_lrhs = HSI_LR[:, :, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                        int(k / downsample_factor):int((k + training_size) / downsample_factor)]
            temp_hrhs = HSI_HR[:, :, j:j + training_size, k:k + training_size]
            with torch.no_grad():
                *_,out = net2(temp_lrhs,temp_hrms)   # ssgt
                # out,ss1,ss2 = net2(temp_lrhs,temp_hrms)   # Fuformer
                # out = net2(temp_lrhs,temp_hrms)  # hsrnet
                # out, out_spat, out_spec, edge_spat1, edge_spat2, edge_spec = net2(temp_lrhs, temp_hrms)   # ssrnet
                assert torch.isnan(out).sum() == 0

                loss_temp = loss_func(out, temp_hrhs.cuda())
                val_loss.update(loss_temp)
                HSI = out.squeeze()
                # 去掉维数为一的维度
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HSI
                index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]

    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon,val_loss




training_size=64
stride=32
downsample_factor=2
# HSI = np.load(r"D:\data\GF5-GF1-new\GF5-GF1\reg_msi.npy")
# MSI = np.load(r"D:\data\GF5-GF1-new\GF5-GF1\reg_pan.npy")
# R = np.load(r"D:\data\GF5-GF1-new\GF5-GF1\R.npy")
# C = np.load(r"D:\data\GF5-GF1-new\GF5-GF1\C.npy")
# HRHSI = np.transpose(HSI, (2, 0, 1))
# HRMSI = np.transpose(MSI, (2, 0, 1))
# R=np.transpose(R, (1,0))
# HSI_LR = Gaussian_downsample(HRHSI, C, downsample_factor)
# MSI_LR = Gaussian_downsample(HRMSI, C, downsample_factor)

#
HSI = np.load(r"Dataset/GF-5/reg_msi.npy")
MSI = np.load(r"Dataset/GF-5/reg_pan.npy")
R = np.load(r"Dataset/GF-5/R.npy")
C = np.load(r"Dataset/GF-5/C.npy")
R = np.transpose(R, (1, 0))
HSI_LR = Gaussian_downsample(np.transpose(HSI, (2, 0, 1)), C, downsample_factor)
LRMSI = Gaussian_downsample(np.transpose(MSI, (2, 0, 1)), C, downsample_factor)
test_HRHSI0 = np.transpose(HSI, (2, 0, 1))[:, 800:, :400]
test_HRMSI0 = LRMSI[:, 800:, :400]
test_LRHSI0 = HSI_LR[:, int(800 / downsample_factor):, :int(400 / downsample_factor)]
# test_HRHSI0 = np.transpose(HSI, (2, 0, 1))
# test_HRMSI0 = LRMSI
# test_LRHSI0 = HSI_LR

val_loss = AverageMeter()
SAM = Loss_SAM()
RMSE = Loss_RMSE()
PSNR = Loss_PSNR()
sam = AverageMeter()
rmse = AverageMeter()
psnr = AverageMeter()




with torch.no_grad():
    test_HRHSI = torch.unsqueeze(torch.Tensor(test_HRHSI0), 0)
    test_HRMSI = torch.unsqueeze(torch.Tensor(test_HRMSI0), 0)
    test_LRHSI = torch.unsqueeze(torch.Tensor(test_LRHSI0), 0)
    time1= time.time()
    prediction, val_loss = reconstruction_fg5(net2, R, test_LRHSI.cuda(),test_HRMSI.cuda(),  test_HRHSI,
                                              downsample_factor, training_size, stride, val_loss)
    Fuse = prediction.cpu().detach().numpy()

time2 = time.time()
print(time2 - time1)
sam.update(SAM(np.transpose(test_HRHSI.squeeze().cpu().detach().numpy(), (1, 2, 0)),
               np.transpose(prediction.squeeze().cpu().detach().numpy(), (1, 2, 0))))
rmse.update(RMSE(test_HRHSI.squeeze().cpu().permute(1, 2, 0), prediction.squeeze().cpu().permute(1, 2, 0)))
psnr.update(PSNR(test_HRHSI.squeeze().cpu().permute(1, 2, 0), prediction.squeeze().cpu().permute(1, 2, 0)))
qnr = qnr_index( 
    np.transpose(prediction.squeeze().cpu().numpy(), (1, 2, 0)),
    np.transpose(test_HRMSI.squeeze().cpu().numpy(), (1, 2, 0)),
    np.transpose(test_LRHSI.squeeze().cpu().numpy(), (1, 2, 0))
)
faker_hyper = np.transpose(Fuse, (1, 2, 0))
print(PSNR(test_HRHSI.squeeze().cpu().permute(1, 2, 0), prediction.squeeze().cpu().permute(1, 2, 0)))
rea= np.transpose(test_LRHSI0, (1, 2, 0))
test_data_path_1 = os.path.join(save_path + "gf5_fak")
test_data_path_2 = os.path.join(save_path+"gf5_rea")
hdf5storage.savemat(test_data_path_1, {'fak': faker_hyper}, format='5')
# hdf5storage.savemat(test_data_path_2, {'rea': rea}, format='5')
print("val loss:",val_loss.avg)
print("val  PSNR:", psnr.avg.cpu().detach().numpy(), "  RMSE:", rmse.avg.cpu().detach().numpy(), "  SAM:", sam.avg,"  QNR:", qnr)
