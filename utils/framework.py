import cv2
import torch
import warnings
import numpy as np

from timm.optim import create_optimizer_v2
from torch.autograd import Variable as V
from timm.scheduler import PlateauLRScheduler, CosineLRScheduler
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

class TrainFramework(object):
    def __init__(self, net, loss, scheduler, training_epochs=500, cooldown_epochs=150, lr=2e-4, weight_decay=1e-2,
                    decay_rate=0.5, patience_t=5, warmup_t=5, warmup_lr_init=5e-6, lr_min=5e-6):
        self.img_id = None
        self.mask = None
        self.img = None
        self.net = net
        self.loss = loss()
        # if torch.cuda.device_count() > 1:
        #     self.net = torch.nn.parallel.DistributedDataParallel(self.net)
        self.optimizer = create_optimizer_v2(self.net, opt="lookahead_AdamW", lr=lr, weight_decay=weight_decay,)
        if scheduler == 'Cosine':
            print('CosineLRScheduler')
            self.scheduler = CosineLRScheduler(self.optimizer, t_initial=training_epochs - cooldown_epochs,
                                                lr_min=lr_min, warmup_t=warmup_t, warmup_lr_init=warmup_lr_init)
        else:
            print('PlateauLRScheduler')
            self.scheduler = PlateauLRScheduler(self.optimizer, decay_rate=decay_rate, patience_t=patience_t,
                                                verbose=False, warmup_t=warmup_t, warmup_lr_init=warmup_lr_init,
                                                lr_min=lr_min, mode='min')
        self.scaler = GradScaler()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(non_blocking=True), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(non_blocking=True), volatile=volatile)

    def optimize(self):
        self.forward()
        with autocast():
            # if self.loss.__class__.__name__ == 'gf_loss':
            #     pred, e1, e2, e3, e4 = self.net.forward(self.img)
            #     loss = self.loss(self.mask, pred, e1, e2, e3, e4)
            # else:
            pred = self.net.forward(self.img)
            loss = self.loss(self.mask, pred)
        # with autocast():
        #     pred = self.net.forward(self.img)
        #     loss = self.loss(self.mask, pred)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return loss.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        # model_dict = self.net.state_dict()
        # checkpoint = {key: value for key, value in torch.load(path).items() if
        #                    (key in model_dict and 'decoder' not in key and 'final' not in key)}
        # model_dict.update(checkpoint)
        # self.net.load_state_dict(model_dict)
        self.net.load_state_dict(torch.load(path))  # , strict=False

    def load_state_dict(self, checkpoint):
        # model_dict = self.net.state_dict()
        # checkpoint = {key: value for key, value in checkpoint.items() if
        #                    (key in model_dict and 'decoder' not in key and 'final' not in key)}
        # model_dict.update(checkpoint)
        # self.net.load_state_dict(model_dict)
        self.net.load_state_dict(checkpoint)  #, strict=False


class TTAFramework(object):
    def __init__(self, net):
        self.net = net

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        BATCHSIZE_PER_CARD = 2
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]
        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)
        img1 = V(torch.from_numpy(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda(non_blocking=True))
        img2 = V(torch.from_numpy(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda(non_blocking=True))
        img3 = V(torch.from_numpy(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda(non_blocking=True))
        img4 = V(torch.from_numpy(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda(non_blocking=True))
        maska = torch.sigmoid(self.net.forward(img1)).squeeze().cpu().data.numpy()
        maskb = torch.sigmoid(self.net.forward(img2)).squeeze().cpu().data.numpy()
        maskc = torch.sigmoid(self.net.forward(img3)).squeeze().cpu().data.numpy()
        maskd = torch.sigmoid(self.net.forward(img4)).squeeze().cpu().data.numpy()
        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]
        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]
        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)
        img1 = V(torch.from_numpy(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda(non_blocking=True))
        img2 = V(torch.from_numpy(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda(non_blocking=True))
        img3 = V(torch.from_numpy(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda(non_blocking=True))
        img4 = V(torch.from_numpy(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda(non_blocking=True))
        maska = torch.sigmoid(self.net.forward(img1)).squeeze().cpu().data.numpy()
        maskb = torch.sigmoid(self.net.forward(img2)).squeeze().cpu().data.numpy()
        maskc = torch.sigmoid(self.net.forward(img3)).squeeze().cpu().data.numpy()
        maskd = torch.sigmoid(self.net.forward(img4)).squeeze().cpu().data.numpy()
        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]
        return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.from_numpy(img5).to('cuda:0',non_blocking=True))
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.from_numpy(img6).cuda(non_blocking=True))
        maska = torch.sigmoid(self.net.forward(img5)).squeeze().cpu().data.numpy()
        maskb = torch.sigmoid(self.net.forward(img6)).squeeze().cpu().data.numpy()
        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]
        return mask3

    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.from_numpy(img5).cuda(non_blocking=True))
        mask = torch.sigmoid(self.net.forward(img5)).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        lab_image = cv2.cvtColor(mask3, cv2.COLOR_GRAY2BGR)
        lab_image = cv2.cvtColor(lab_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_image)
        l_new = cv2.add(l, np.ones_like(l) * 10)
        l_new = np.clip(l_new, 0, 255)
        lab_image_new = cv2.merge((l_new, a, b))
        # image_new = cv2.cvtColor(lab_image_new, cv2.COLOR_LAB2BGR)
        # image_new = cv2.cvtColor(image_new, cv2.COLOR_BGR2GRAY)

        return lab_image_new

    def load(self, path):
        self.net.load_state_dict(torch.load(path))  # , strict=False
