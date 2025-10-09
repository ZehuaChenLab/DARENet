import os
import cv2
import time
import errno
import numpy as np
from tqdm import tqdm
from utils import Config
from utils.logger import Logger
from utils.metrics import Evaluator
from utils.framework import TTAFramework

from networks.unet import Unet
from networks.carnet import CARNet
from networks.rcfsnet import RCFSNet
from networks.dlinknet import DLinkNet
from networks.nllinknet import NLlinkNet
from networks.cmlformer import CMLFormer
from networks.deeplabv3plus import DeepLabv3_plus
from networks.CMTFNet import CMTFNet
from networks.CARENet import CARENet
from networks.baseline import Baseline
from networks.baseline_mixer_dskip import Baseline_Mixer_Dskip
from MyNet.FuseExperiment import Baseline_Ex
from MyNet.FuseExperiment2 import Baseline_Ex2
from MyNet.NewNet import NewNet
from MyNet.ConvRoad import ConvRoad
from MyNet.MobileRoadNet import MobileRoadNet
from MyNet.LID2Mamba import LID2Mamba,LID2MambaMobile
from MyNet.GLLANet import GLLANet
from networks.CPSSNet import CPSSNet
from MyNet.GLLAMamba import GLLAMamba
from networks.lmz import DARENet

def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

if __name__ == '__main__':
    model_dict = {'UNet': Unet, 'DLinkNet': DLinkNet, 'NLlinkNet': NLlinkNet, 'DeepLabv3_plus': DeepLabv3_plus,
                'CARNet':CARNet, 'RCFSNet':RCFSNet, 'CMLFormer':CMLFormer,
                'Baseline': Baseline, 'Baseline_Mixer_Dskip': Baseline_Mixer_Dskip,'Baseline_Ex':Baseline_Ex,'CMTFNet':CMTFNet,'CARENet':CARENet,
                  'NewNet':NewNet,'MobileRoadNet':MobileRoadNet,'LID2Mamba':LID2Mamba,'LID2MambaMobile':LID2MambaMobile,'ConvRoad':ConvRoad,
                  'GLLANet':GLLANet,'CPSSNet':CPSSNet,'GLLAMamba':GLLAMamba,'DARENet':DARENet}
    model = model_dict[Config.MODEL_NAME]
    if Config.DATASET.split('/')[-1] == 'ROAD':
        net = model(img_size=1024).cuda()
    elif Config.DATASET.split('/')[-1] == 'Mas':
        net = model(img_size=1500).cuda()
    elif Config.DATASET.split('/')[-1] == 'spacenet':
        net = model(img_size=1024).cuda()
        # net = model().cuda()
    source = Config.DATASET + '/valid/'
    model_name = Config.MODEL_NAME
    target = Config.DATASET.split('/')[-1]
    if Config.WEIGHT == '':
        # 0.2800216868519783
        weight = './weights/best_weights/'+ model_name + '_' + target + '_0.17260034452985834'+'_best_model.pth'
        # weight = './weights/best_weights/' + model_name + '_' + target + '_best_model.pth'
    else:
        weight = Config.WEIGHT
    log_name = model_name + '_' + target
    threshold_list = [x / 10 for x in range(5, 50, 1)]

    if target == '':
        target = './submits/' + model_name + '/'
    else:
        target = './submits/' + model_name + '/' + target + '/'
    if not os.path.exists(target):
        try:
            os.makedirs(target)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
    dataset = source.split('/')[-3]
    imagelist = list(filter(lambda x: x.find('sat') != -1, os.listdir(source)))
    if dataset == 'ROAD':
        testlist = list(map(lambda x: x[:-8], imagelist))
        img_suf = '_sat.jpg'
        mask_suf = '_mask.png'
    elif dataset == 'Mas':
        testlist = list(map(lambda x: x[:-9], imagelist))
        img_suf = '_sat.tiff'
        mask_suf = '_mask.tif'
    elif dataset == 'spacenet':
        testlist = list(map(lambda x: x[:-8], imagelist))
        img_suf = '_sat.png'
        mask_suf = '_mask.png'
    myevaluator = Evaluator(2)

    solver = TTAFramework(net)
    solver.load(weight)
    best_f1 = 0
    best_iou = 0
    best_threshold = 1.0
    mask_list = []
    gt_list = []
    mkdir('./logs/test/')
    mylog = Logger('./logs/test/' + log_name + '.log')
    for i, name in tqdm(enumerate(testlist), total=len(testlist), desc="Testing "):
        mask_list.append(solver.test_one_img_from_path(source + name + img_suf))
        ground_truth = cv2.imread(source + name + mask_suf, cv2.IMREAD_GRAYSCALE)
        gt_list.append(ground_truth)
    if Config.THRESHOLD is not None:
        mylog.write("Used THRESHOLD in the Config" + '\n')
        best_threshold = Config.THRESHOLD
    else:
        for threshold in threshold_list:
            loop_eval = tqdm(enumerate(zip(mask_list, gt_list, testlist)), total=len(mask_list), desc='threshold='+str(threshold))
            for index, (mask, ground_truth, name) in loop_eval:
                new_mask = np.copy(mask)
                new_mask[new_mask > threshold] = 255
                new_mask[new_mask <= threshold] = 0
                new_mask = np.array(new_mask, np.float32) / 255.0
                gt_image = np.array(ground_truth, np.float32) / 255.0
                myevaluator.update(gt_image, new_mask)
            IoU = myevaluator.Intersection_over_Union()
            myevaluator.Pixel_Precision()
            myevaluator.Pixel_Recall()
            f1 = myevaluator.Pixel_F1()
            mylog.write('************************' + '\n')
            mylog.write('Threshold:       {}'.format(threshold) + '\n')
            mylog.write('IoU_Score:       {}'.format(IoU) + '\n')
            mylog.write('F1_Score:        {}'.format(f1) + '\n')
            if IoU + f1 > best_iou + best_f1:
                best_f1 = f1
                best_iou = IoU
                best_threshold = threshold
            myevaluator.reset()
            time.sleep(0.1)
    for mask, ground_truth, name in zip(mask_list, gt_list, testlist):
        mask[mask > best_threshold ] = 255
        mask[mask <= best_threshold ] = 0
        new_mask = np.copy(mask)
        new_mask = np.array(new_mask, np.float32) / 255.0
        ground_truth = np.array(ground_truth, np.float32) / 255.0
        myevaluator.update(ground_truth, new_mask)
        mylog.write('************************' + '\n')
        mylog.write('name:            {}'.format(name) + '\n')
        mylog.write('Accuracy:        {}'.format(myevaluator.img_metricser.img_accuracy) + '\n')
        mylog.write('Precision:       {}'.format(myevaluator.img_metricser.img_precision) + '\n')
        mylog.write('Recall:          {}'.format(myevaluator.img_metricser.img_recall) + '\n')
        mylog.write('F1_Score:        {}'.format(myevaluator.img_metricser.img_f1) + '\n')
        mylog.write('IoU_Score:       {}'.format(myevaluator.img_metricser.img_IoU) + '\n')
        mask = mask[:, :, None]
        mask = np.concatenate([mask, mask, mask], axis=2)
        cv2.imwrite(target + name + mask_suf, mask.astype(np.uint8))
    mylog.write('************************' + '\n')
    mylog.write('Threshold:       {}'.format(best_threshold) + '\n')
    mylog.write('Accuracy:        {}'.format(myevaluator.Pixel_Accuracy()) + '\n')
    mylog.write('Precision:       {}'.format(myevaluator.Pixel_Precision()) + '\n')
    mylog.write('Recall:          {}'.format(myevaluator.Pixel_Recall()) + '\n')
    mylog.write('F1_Score:        {}'.format(myevaluator.Pixel_F1()) + '\n')
    mylog.write('IoU_Score:       {}'.format(myevaluator.Intersection_over_Union()) + '\n')
    mylog.write('mIoU_Score:      {}'.format(myevaluator.mean_Intersection_over_Union()) + '\n')
    mylog.close()
