import os
import errno
import torch

from time import time
from tqdm import tqdm
from utils import Config_space
from utils.saver import Saver
from utils.logger import Logger
from torch.backends import cudnn
from utils.data1 import ImageFolder
from utils.emailer import send_email
from utils.loss import dice_bce_loss
from utils.framework import TrainFramework

from networks.unet import Unet
from networks.carnet import CARNet
from networks.rcfsnet import RCFSNet
from networks.dlinknet import DLinkNet
from networks.nllinknet import NLlinkNet
from networks.cmlformer import CMLFormer
from networks.llflinknet import LLF_LinkNet
from networks.deeplabv3plus import DeepLabv3_plus
from networks.CMTFNet import CMTFNet
from networks.samba import Samba
from MyNet.FuseExperiment import Baseline_Ex
from MyNet.NewNet import NewNet
from MyNet.FuseExperiment2 import Baseline_Ex2
from networks.baseline import Baseline
from networks.baseline_mixer_dskip import Baseline_Mixer_Dskip
from networks.CARENet import CARENet

from MyNet.MobileRoadNet import MobileRoadNet
from MyNet.LID2Mamba import LID2Mamba,LID2MambaMobile
from MyNet.GLLANet import GLLANet
from networks.CPSSNet import CPSSNet

def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def log_tqdm_writer(msg, logger, mytqdm):
    logger.write(msg+ '\n')
    mytqdm.write(msg)



if __name__ == "__main__":
    print(torch.__version__)
    cudnn.benchmark = True

    model_dict = {'UNet': Unet, 'DLinkNet': DLinkNet, 'NLlinkNet': NLlinkNet, 'DeepLabv3_plus': DeepLabv3_plus,
                  'CARNet':CARNet, 'RCFSNet':RCFSNet, 'CMLFormer':CMLFormer,
                  'Baseline': Baseline, 'Baseline_Mixer_Dskip': Baseline_Mixer_Dskip,'Baseline_Ex':Baseline_Ex,'Baseline_Ex2':Baseline_Ex2,
                  'CMTFNet':CMTFNet,'Samba':Samba,'CARENet':CARENet,'NewNet':NewNet,'MobileRoadNet':MobileRoadNet,'LID2Mamba':LID2Mamba,
                  'LID2MambaMobile':LID2MambaMobile,'GLLANet':GLLANet,'CPSSNet':CPSSNet}
    loss_dict = {'dice_bce_loss': dice_bce_loss}

    model = model_dict[Config_space.MODEL_NAME]
    # dataset = Config_space.DATASET + '/train/'
    load = Config_space.WEIGHT
    name = Config_space.MODEL_NAME + '_' + Config_space.DATASET.split('/')[-1]

    Loss = loss_dict[Config_space.LOSS]

    if Config_space.DATASET.split('/')[-1] == 'ROAD':
        # net = model(img_size=1024).cuda()
        net = model(img_size=1024).cuda()
    elif Config_space.DATASET.split('/')[-1] == 'Mas':
        net = model(img_size=1500).cuda()
        # net = model().cuda()
    elif Config_space.DATASET.split('/')[-1] == 'CHN6-CUG':
        net = model(img_size=512).cuda()
        # net = model().cuda()
    elif Config_space.DATASET.split('/')[-1] == 'spacenet':
        # net = model(img_size=1024).cuda()
        net = model().cuda()


    solver = TrainFramework(net, Loss, Config_space.SCHEDULER, Config_space.TOTAL_EPOCH, Config_space.COOLDOWN_EPOCH,
                            Config_space.INIT_LEARNING_RATE, Config_space.WEIGHT_DECAY, Config_space.DECAY_RATE, Config_space.PATIENCE_T,
                            Config_space.WARMUP_T, Config_space.WARMUP_LR_INIT, Config_space.MIN_LEARNING_RATE)
    mysaver = Saver()
    batchsize = torch.cuda.device_count() * 4
    dataset = ImageFolder(root_path='/home/smart/Road/dataset/spacenet',datasets ='spacenet')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=16,
        pin_memory=True)

    if Config_space.RESUME:
        if os.path.isfile("./weights/last_weights/" + name + "_last_model.pth"):
            print("Resume from checkpoint...")
            mysaver.checkpoint = torch.load("./weights/last_weights/" + name + "_last_model.pth")
            mysaver.load_checkpoint(solver)
            init_epoch = mysaver.checkpoint['epoch'] + 1
            init_time = mysaver.checkpoint['total_time']
            train_epoch_best_loss = mysaver.checkpoint['train_epoch_best_loss']
            no_optim = mysaver.checkpoint['no_optim']
            print("====>loaded checkpoint (epoch:{})".format(mysaver.checkpoint['epoch']))
        else:
            print("====>no checkpoint found.")
            exit()
    else:
        init_epoch = 1
        init_time = 0
        no_optim = 0
        train_epoch_best_loss = 100.
        if load != '':
            print('Loading...')
            solver.load(load)
            if Config_space.LOSS_UPDATE:
                train_epoch_best_loss = float(load.split('/')[-1].split('_')[-3])

    mkdir('./logs/train/')
    mylog = Logger('./logs/train/' + name + '.log')
    total_time = init_time
    total_epoch = init_epoch
    mid_loss = Config_space.INIT_MID_LOSS
    for epoch in range(init_epoch, Config_space.TOTAL_EPOCH + 1):
        tic = time()
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        solver.loss.update()
        num_updates = epoch * len(data_loader)
        loop_train = tqdm(enumerate(data_loader), total=len(data_loader))
        for index, (img, mask) in loop_train:
            num_updates += 1
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            solver.scheduler.step_update(num_updates)
            train_epoch_loss += train_loss
            loop_train.set_description(f'Epoch [{epoch}/{Config_space.TOTAL_EPOCH}]')
            loop_train.set_postfix(loss=train_loss,lr=solver.optimizer.param_groups[0]["lr"])
        train_epoch_loss /= len(data_loader_iter)
        solver.optimizer.sync_lookahead()
        solver.scheduler.step(epoch + 1, metric=train_epoch_loss)
        total_epoch += 1
        total_time = int(time() - tic) + total_time
        log_tqdm_writer('************************', mylog, tqdm)
        log_tqdm_writer('epoch: {}      time: {}'.format(epoch, total_time), mylog, tqdm)
        log_tqdm_writer('learning_rate: {:.2e}'.format(solver.optimizer.param_groups[0]["lr"]), mylog, tqdm)
        log_tqdm_writer('train_loss:    {:.6f}'.format(train_epoch_loss), mylog, tqdm)
        for k, v in solver.loss.get_loss().items():
            log_tqdm_writer(k + ':     {:.6f}'.format(v / len(data_loader_iter)), mylog, tqdm)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            if mid_loss - train_epoch_loss >= Config_space.LOSS_GAP:
                mkdir('./weights/mid_weights/')
                solver.save("./weights/mid_weights/" + name + '_' + str(train_epoch_loss) + "_mid_model.pth")
                mid_loss = train_epoch_loss
            train_epoch_best_loss = train_epoch_loss
            mkdir('./weights/best_weights/')
            solver.save("./weights/best_weights/" + name + "_best_model.pth")
            mysaver.best_weight = solver.net.state_dict()
            log_tqdm_writer("!!--Best Model has Update--!!", mylog, tqdm)

        path_checkpoint = "./weights/last_weights/" + name + "_last_model.pth"
        mkdir('./weights/last_weights/')
        mysaver.save_checkpoint(solver, epoch, train_epoch_best_loss, total_time, no_optim, path_checkpoint)
        if no_optim > Config_space.NUM_EARLY_STOP:
            break
        mylog.flush()

    mysaver.save_best_weight("./weights/best_weights/" + name + '_' + str(train_epoch_best_loss) + "_best_model.pth")
    subject = name + " training is complete!!!"
    epoch_content = 'total_epoch:' + str(total_epoch) + '\ntotal_time:' + str(total_time) + '\n'
    loss_content = 'last_train_loss:' + str(train_epoch_best_loss) + '\n'
    content = subject + '\n' + epoch_content + loss_content
    send_email(subject, content)
    mylog.write(content)
    mylog.close()
