import time
import torch
from thop import profile, clever_format
from Model.ASSR.VGSR import VGSR_NET
# from Models.DSP import  DSPNet

# from Models.OTIAS.otias import otias_c_x8
# from Models.VGSRLight.VGSR import VGSR_NET
# from  Models.BiSER.BiSER_Netv4 import BiSER
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def test_flops(model,hsi,msi):
    flops, params, DIC = profile(model, inputs=(hsi, msi), ret_layer_info=True)
    print(flops, params)
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("总参数量: {:.2f}M".format(total_params / 1e6))
    out = model(hsi, msi)
    print("out shape:",out.shape)
    for i in range(10):
        time1 = time.time()
        out = model(hsi, msi)
        time2 = time.time()
        print(time2 - time1)
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("测试FLops")
    model = VGSR_NET(31,3,64).cuda()
    # model = DSPNet(31,3).cuda()
    # model = otias_c_x8(n_select_bands=3,n_bands = 31,feat_dim=128,guide_dim=128,sz = 64).cuda()
    a,b=64,64
    hsi = torch.randn((1,31,a//8,b//8)).cuda()
    msi = torch.randn((1,3,a,b)).cuda()
    test_flops(model,hsi,msi)
    # model = SSHNET(31,3).cuda()
    # test_flops(model,hsi,msi)



