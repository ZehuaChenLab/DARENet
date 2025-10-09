import os
import cv2
import torch
import errno
import numpy as np
from tqdm import tqdm
from utils import Config
from utils.visualizer import vis_cam
from torch.autograd import Variable as V

from networks.unet import Unet
from networks.carnet import CARNet
from networks.rcfsnet import RCFSNet
from networks.dlinknet import DLinkNet
from networks.nllinknet import NLlinkNet
from networks.cmlformer import CMLFormer
from networks.deeplabv3plus import DeepLabv3_plus

from networks.baseline import Baseline
from MyNet.LID2Mamba import LID2Mamba
# from networks.baseline_mixer import Baseline_Mixer
# from networks.baseline_dskip import Baseline_Dskip
# from networks.baseline_mixer_dskip import Baseline_Mixer_Dskip

from MyNet.FuseExperiment import Baseline_Ex
from networks.CARENet import CARENet
from networks.CARENet import FEM
from networks.CARENet import DBSSM

# MoblieRoadNet
from MyNet.MobileRoadNet import MobileRoadNet
from MyNet.MobileRoadNet import ECA
from MyNet.MobileRoadNet import ESA
from MyNet.MobileRoadNet import MSFFN


if __name__ == "__main__":
    model_dict = {'UNet': Unet, 'DeepLabv3_plus': DeepLabv3_plus, 'DLinkNet': DLinkNet, 'NLlinkNet': NLlinkNet,
                  'CARNet': CARNet, 'RCFSNet': RCFSNet, 'CMLFormer': CMLFormer,
                  'Baseline': Baseline, 'FEM':FEM,'DBSSM':DBSSM,'Baseline_Ex':Baseline_Ex,'CARENet':CARENet,'LID2Mamba':LID2Mamba,'MobileRoadNet': MobileRoadNet}
    model = model_dict[Config.MODEL_NAME]
    if Config.DATASET.split('/')[-1] == 'ROAD':
        net = model(img_size=1024).cuda()
    elif Config.DATASET.split('/')[-1] == 'Mas':
        net = model(img_size=1500).cuda()
    source = Config.DATASET + '/valid/'
    model_name = Config.MODEL_NAME
    target = Config.DATASET.split('/')[-1]
    if Config.WEIGHT == '':
        weight = './weights/best_weights/' + model_name + '_' + target +'_0.15314197341958397'+ '_best_model.pth'
        # weight = './weights/bast_weights/' + model_name + '_' + target  + '_best_model.pth'
    else:
        weight = Config.WEIGHT
    log_name = model_name + '_' + target

    if target == '':
        target = './visualize/' + model_name + '/'
    else:
        target = './visualize/' + model_name + '/' + target + '/'
    if not os.path.exists(target):
        try:
            os.makedirs(target)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    dataset = source.split('/')[-3]
    imagelist = list(filter(lambda x: x.find('sat') != -1, os.listdir(source)))
    vislist = []
    img_suf = ''
    mask_suf = ''
    if dataset == 'ROAD':
        vislist = list(map(lambda x: x[:-8], imagelist))
        img_suf = '_sat.jpg'
        mask_suf = '_mask.png'
    elif dataset == 'Mas':
        vislist = list(map(lambda x: x[:-9], imagelist))
        img_suf = '_sat.tiff'
        mask_suf = '_mask.tif'

    net.load_state_dict(torch.load(weight))
    net.eval()
    # target_layers = [[net.fussion.down1.mlp.fc2[1]], [net.fussion.down2.mlp.fc2[1]], [net.fussion.down3.mlp.fc2[1]],
    #                 [net.fussion.up3.mlp.fc2[1]], [net.fussion.up2.mlp.fc2[1]], [net.fussion.up1.mlp.fc2[1]],
    #                 [net.dblock.local_unit.proj[3]], [net.dblock.global_unit.proj[3]], [net.dblock.mlp.fc2[1]]]
    
    """
    _visualize_
    """
    
    target_layers = [[net.c1.mlp.fc2[1]],[net.c2.mlp.fc2[1]],[net.c3.mlp.fc2[1]],[net.c4.mlp.fc2[1]]]
    
    """
    _visualize_
    """
    
    # target_layers = [[net.s4.convs[3].mlp.fc2[1]]]
    #
                     # net.s4.convs[3].mlp.fc2[1]]
    for i, name in tqdm(enumerate(vislist), total=len(vislist), desc="Visualizing "):
        canvas_img = np.array(cv2.imread(source + name + img_suf))
        gt_image = cv2.imread(source + name + mask_suf, cv2.IMREAD_GRAYSCALE)
        gt_image = np.array(gt_image, np.float32) / 255.0
        net_input = canvas_img.astype(np.float32) / 255.0 * 3.2 - 1.6
        net_input = V(torch.from_numpy(net_input).cuda(non_blocking=True)).unsqueeze(dim=0)
        net_input = net_input.permute(0, 3, 1, 2).contiguous()
        for j in range(0,len(target_layers)):
            mask = vis_cam(net, target_layers=target_layers[j], net_input=net_input, canvas_img=canvas_img,
                           gt_image=gt_image)
            if not os.path.exists(target + str(j) + '/'):
                os.makedirs(target + str(j) + '/')
            cv2.imwrite(target + str(j) + '/' + name + mask_suf, mask.astype(np.uint8))
        # mask = vis_cam(net, target_layers=target_layers, net_input=net_input, canvas_img=canvas_img,
        #                    gt_image=gt_image)
        # cv2.imwrite(target + name + mask_suf, mask.astype(np.uint8))
