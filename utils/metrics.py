"""
Based on https://github.com/mj129/CoANet
"""
import numpy as np

class img_metricser(object):
    def __init__(self, num_class):
        self.img_f1 = None
        self.img_IoU = None
        self.img_recall = None
        self.img_accuracy = None
        self.img_precision = None
        self.num_class = num_class
        self.img_confusion_matrix = np.zeros((self.num_class,) * 2)

    def img_Pixel_Accuracy(self):
        Acc = np.diag(self.img_confusion_matrix).sum() / self.img_confusion_matrix.sum()
        return Acc

    def img_Pixel_Precision(self):
        if self.img_confusion_matrix[1, 1] == 0:
            self.img_precision = 0
        else:
            self.img_precision = self.img_confusion_matrix[1, 1] / (self.img_confusion_matrix[1, 1] + self.img_confusion_matrix[0, 1])
        return self.img_precision

    def img_Pixel_Recall(self):
        if self.img_confusion_matrix[1, 1] == 0:
            self.img_recall = 0
        else:
            self.img_recall = self.img_confusion_matrix[1, 1] / (self.img_confusion_matrix[1, 1] + self.img_confusion_matrix[1, 0])
        return self.img_recall

    def img_Pixel_F1(self):
        if self.img_precision == 0 or self.img_recall == 0:
            f1 = 0
        else:
            f1 = 2 * self.img_precision * self.img_recall / (self.img_precision + self.img_recall)
        return f1

    def img_Intersection_over_Union(self):
        img_IoU = self.img_confusion_matrix[1, 1] / (self.img_confusion_matrix[1, 1] + self.img_confusion_matrix[1, 0] + self.img_confusion_matrix[0, 1] + 1e-10)
        return img_IoU

    def img_update(self):
        self.img_accuracy = self.img_Pixel_Accuracy()
        self.img_recall = self.img_Pixel_Recall()
        self.img_precision = self.img_Pixel_Precision()
        self.img_f1 = self.img_Pixel_F1()
        self.img_IoU = self.img_Intersection_over_Union()



class Evaluator(object):
    def __init__(self, num_class):
        self.recall = None
        self.precision = None
        self.num_class = num_class
        self.img_metricser = img_metricser(num_class)
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Precision(self):
        self.precision = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[0, 1])
        return self.precision

    def Pixel_Recall(self):
        self.recall = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0])
        return self.recall

    def Pixel_F1(self):
        f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        return f1

    def Intersection_over_Union(self):
        IoU = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0] + self.confusion_matrix[0, 1] + 1e-10)
        return IoU

    def mean_Intersection_over_Union(self):
        mIoU = 0.5*(self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0] +
                                             self.confusion_matrix[0, 1] + 1e-10) + self.confusion_matrix[0, 0] /
                   (self.confusion_matrix[0, 0] + self.confusion_matrix[1, 0] + self.confusion_matrix[0, 1] + 1e-10))

        return mIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def update(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        # self.img_confusion_matrix = self._generate_matrix(gt_image, pre_image)
        self.img_metricser.img_confusion_matrix = self._generate_matrix(gt_image, pre_image)
        self.confusion_matrix += self.img_metricser.img_confusion_matrix
        self.img_metricser.img_update()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
