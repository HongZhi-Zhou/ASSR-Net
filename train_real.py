import torch
from scipy.io import loadmat

from calculate_metrics import Loss_SAM, Loss_RMSE, Loss_PSNR
# from ablation.Tfusion_v1_v2 import Cross_Guide_Fusion
# from models.Tfusion_v1_v4 import Cross_Guide_Fusion
# from ablation2.nomask import Cross_Guide_Fusion
# from model_finally.finally2_3stage import Cross_Guide_Fusion
# from model_finally.MF_resnet import Cross_Guide_Fusion
# from model_finally.resnet_mask import Cross_Guide_Fusion
# from SSGTN.SSGTN_noCSA import Cross_Guide_Fusion
# from models.DBIN import FusionNet
# from SSGTN.SSGTN_gf5 import Cross_Guide_Fusion
# from models.Fuformer import MainNet
# from models.SSRnet import SSRNET
from Model.ASSR.VGSRgf import VGSR_NET
# from Models.DSP import *
# from Train_MIMO.Model import *
# from models.HSRnet import RGBNet
# from model.s38bgf import *
from train_dataloader import *
from torch import nn
from tqdm import tqdm
import time
import pandas as pd
import torch.utils.data as data
import torch.nn.functional as F

from utils import create_F, fspecial
import math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# from Train_MIMO.Model import Net_s

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(torch.abs(diff))
        return loss
        
def mkdir(path):

    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("训练文件夹为：{}".format(path))
    else:
        print('已存在{}'.format(path))

def create_fully_connected_adj(B, H, W, device):
    N = H * W
    adj = torch.ones(B, N, N, device=device) / (N - 1)
    return adj


if __name__ == '__main__':
    # 路径参数
    root=os.getcwd()+"/PKL/GaoFen"
    model_name='VGSR'
    mkdir(os.path.join(root,model_name))
    ori_list=os.listdir(os.path.join(root,model_name))
    current_list=[]
    for i in ori_list:
        if len(i)<=2:
            current_list.append(i)

    del ori_list



    current_list.append('0')
    int_list = [int(x) for x in current_list]
    train_value = max(int_list)+1
    model_name=os.path.join(model_name,str(train_value))

    # data_name='harvard'
    # path1 = 'D:\data\Harvard\harvard_train/'
    # path2 = 'D:\data\Harvard\harvard_test/'

    
    training_size=64
    stride=32
    downsample_factor=2
    data_name='gf5'
    HSI = np.load(r"Dataset/GF-5/reg_msi.npy")
    MSI = np.load(r"Dataset/GF-5/reg_pan.npy")
    R = np.load(r"Dataset/GF-5/R.npy")
    C = np.load(r"Dataset/GF-5/C.npy")
    HRHSI = np.transpose(HSI, (2, 0, 1))
    HRMSI = np.transpose(MSI, (2, 0, 1))
    R=np.transpose(R, (1,0))
    HSI_LR = Gaussian_downsample(HRHSI, C, downsample_factor)
    MSI_LR = Gaussian_downsample(HRMSI, C, downsample_factor)
    _,train_w,train_h=HSI_LR.shape


    print("训练数据处理完成")


    # 训练参数
    loss_func = nn.L1Loss().cuda()
    # loss_func = nn.MSELoss(reduction='mean')



    stride1 = 32
    LR = 2e-4
    EPOCH =2000
    weight_decay=1e-8   # 我的模型是1e-8
    BATCH_SIZE = 16
    num = 20
    psnr_optimal = 50
    rmse_optimal = 1.5

    test_epoch=100
    val_interval = 50           # 每隔val_interval epoch测试一次
    checkpoint_interval = 25
    # maxiteration = math.ceil(((512 - training_size) // stride + 1) ** 2 * num / BATCH_SIZE) * EPOCH
    maxiteration = math.ceil(
         ((train_w - training_size) // stride + 1) * ((train_h - training_size) // stride + 1) / BATCH_SIZE) * EPOCH
    print("maxiteration：", maxiteration)

    # warm_lr_scheduler
    decay_power = 1.5
    init_lr2 = 2e-4
    init_lr1 = 2e-4 / 10
    min_lr=0
    warm_iter = math.floor(maxiteration / 40)

    # 创建方法名字的文件
    path=os.path.join(root,model_name)
    mkdir(path)  # 创建文件夹
    # 创建训练记录文件
    pkl_name=data_name+'_pkl'
    pkl_path=os.path.join(path,pkl_name)      # 模型保存路径
    os.makedirs(pkl_path)      # 创建文件夹
    # 创建excel
    df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss','val_loss','val_rmse', 'val_psnr', 'val_sam', 'val_qnr'])  # 列名
    excel_name=data_name+'_record.csv'
    excel_path=os.path.join(path,excel_name)
    df.to_csv(excel_path, index=False)

    train_data=RealDATAProcess(HSI_LR,MSI_LR,training_size, stride, downsample_factor,C)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    criterion_fft = fftLoss().cuda()

    # cnn = Cross_Guide_Fusion(31,training_size,training_size,1).cuda()
    # cnn =FusionNet().cuda()
    # cnn=VSR_CAS(channel0=31, factor=8, P=torch.Tensor(R), patch_size=training_size).cuda()
    cnn = VGSR_NET(150,4,64).cuda()
    # cnn = DSPNetgf(150,4).cuda()
    # cnn = Net_sgf(150,4).cuda()
    # cnn = Cross_Guide_Fusion(150, training_size, training_size,1).cuda()
    # cnn=MainNet().cuda()
    # cnn=SSRNET('SSRNET',2,1,150).cuda()
    # 模型初始化
    for m in cnn.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    # optimizer = torch.optim.Adam([{'params': cnn.parameters(), 'initial_lr': 1e-1}], lr=LR,betas=(0.2, 0.999),weight_decay=weight_decay)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR,betas=(0.9, 0.999),weight_decay=weight_decay)
    # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor = 0.96, patience = 3,
    #                                                      verbose = True, threshold = 0.00001, threshold_mode ='abs', cooldown = 5, min_lr = 0, eps = 1e-08)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxiteration, eta_min=1e-8, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 300, 700], 0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=500,
    #                                                gamma=0.5)   # Fuformer
    start_epoch = 0
    # resume = True
    resume = False
    path_checkpoint = "checkpoints/500_epoch.pkl"  # 断点路径

    # start_step=0
    if resume:
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        cnn.load_state_dict(checkpoint['model_state_dict'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        # start_step=start_epoch -1
        # scheduler.last_epoch = start_epoch +1 # 设置学习率的last_epoch
        psnr_optimal = checkpoint['psnr_optimal']
        rmse_optimal = checkpoint['rmse_optimal']



    # step = start_step
    step=0   # warm_lr_scheduler要用
    for epoch in range(start_epoch+1, EPOCH+1):
        cnn.train()
        loss_all = []
        loop = tqdm(train_loader, total=len(train_loader))
        for a1, a2, a3 in loop:
            # print(a1.shape,a2.shape,a3.shape)
            # lr = warm_lr_scheduler(optimizer, init_lr1, init_lr2, min_lr,warm_iter, step,
            #                        lr_decay_iter=1, max_iter=maxiteration, power=decay_power)
            step = step + 1
            lr = optimizer.param_groups[0]['lr']
            # output = cnn(a2.cuda(),a3.cuda())     # SSGT
            # output,ss1,ss2 = cnn(a3.cuda(),a2.cuda())   # Fuformer
            output,_ = cnn(a3.cuda(), a2.cuda(),True)   # hsrnet
            loss2 = loss_func(output, a1.cuda())
            loss1 = loss_func(_,a1.cuda())
            loss = 0.8*loss2+0.2*loss1
            # loss = loss2
            loss_temp = loss

            # output_X1, output_X2, output_X = cnn(a3.cuda(),a2.cuda(),False)
            # a1_X = a1.cuda()
            # a1_X2 = F.interpolate(a1_X, scale_factor=0.5)
            # a1_X1 = F.interpolate(a1_X2, scale_factor=0.5)
            # loss_fft = criterion_fft(output_X, a1_X) + criterion_fft(output_X2, a1_X2) + criterion_fft(output_X1, a1_X1)
            # loss1 = loss_func(output_X, a1_X) + loss_func(output_X1, a1_X1) + loss_func(output_X2, a1_X2)
            # loss = loss1 + 0.01 * loss_fft
            # loss_temp = loss
            
            loss_all.append(np.array(loss_temp.detach().cpu().numpy()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
            loop.set_postfix({'loss': '{0:1.8f}'.format(np.mean(loss_all)), "lr": '{0:1.8f}'.format(lr)})
            scheduler.step()





        if ((epoch % val_interval == 0) and (epoch>=test_epoch) ) or epoch==1:
            cnn.eval()
            val_loss=AverageMeter()
            SAM = Loss_SAM()
            RMSE = Loss_RMSE()
            PSNR = Loss_PSNR()
            sam = AverageMeter()
            rmse = AverageMeter()
            psnr = AverageMeter()

            with torch.no_grad():
                # img1 = img1 / img1.max()
                test_HRHSI = torch.unsqueeze(torch.Tensor(HRHSI),0)
                test_HRMSI =torch.unsqueeze(torch.Tensor(MSI_LR),0)
                test_LRHSI=torch.unsqueeze(torch.Tensor(HSI_LR),0)


                prediction,val_loss = reconstruction_fg5(cnn, R, test_LRHSI.cuda(), test_HRMSI.cuda(), test_HRHSI,downsample_factor, training_size, stride1,val_loss)
                # print(Fuse.shape)
                prediction=torch.round(prediction*255)/255.0
                sam.update(SAM(np.transpose(test_HRHSI.squeeze().cpu().detach().numpy(),(1, 2, 0)),np.transpose(prediction.squeeze().cpu().detach().numpy(),(1, 2, 0))))
                rmse.update(RMSE(test_HRHSI.squeeze().cpu().permute(1,2,0),prediction.squeeze().cpu().permute(1,2,0)))
                psnr.update(PSNR(test_HRHSI.squeeze().cpu().permute(1,2,0),prediction.squeeze().cpu().permute(1,2,0)))

            
            # if  epoch == 1:
                # torch.save(cnn.state_dict(),pkl_path +'/'+ str(epoch) + 'EPOCH' + '_PSNR_best.pkl')
            torch.save(cnn.state_dict(), pkl_path + '/' + str(epoch) + 'EPOCH' + '.pkl')
            # if torch.abs(psnr_optimal-psnr.avg)<0.15:
                # torch.save(cnn.state_dict(), pkl_path + '/' + str(epoch) + 'EPOCH' + '_PSNR_best.pkl')
            # if psnr.avg > psnr_optimal:
                # psnr_optimal = psnr.avg
# 
            # if torch.abs(rmse.avg-rmse_optimal)<0.15:
                # torch.save(cnn.state_dict(),pkl_path +'/'+ str(epoch) + 'EPOCH' + '_RMSE_best.pkl')
            # if rmse.avg < rmse_optimal:
                # rmse_optimal = rmse.avg



            print("val  PSNR:",psnr.avg.cpu().detach().numpy(), "  RMSE:", rmse.avg.cpu().detach().numpy(), "  SAM:", sam.avg,"val loss:", val_loss.avg.cpu().detach().numpy())
            val_list = [epoch, lr,np.mean(loss_all),val_loss.avg.cpu().detach().numpy(),rmse.avg.cpu().detach().numpy(), psnr.avg.cpu().detach().numpy(), sam.avg]

            val_data = pd.DataFrame([val_list])
            val_data.to_csv(excel_path,mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
            time.sleep(0.1)
