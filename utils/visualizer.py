import cv2
import torch
import numpy as np

from utils.loss import dice_bce_loss
from torch.autograd import Variable as V
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform,
        )

    def __call__(
            self,
            input_tensor: torch.Tensor,
            targets=None,
            aug_smooth: bool = True,
            eigen_smooth: bool = False,
    ) -> np.ndarray:
        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor, targets, eigen_smooth)

    def forward(
            self, input_tensor: torch.Tensor, targets, eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        if self.uses_gradients:
            self.model.zero_grad()
            loss = dice_bce_loss()
            targets = V(torch.from_numpy(targets)).cuda(non_blocking=True)
            loss = loss(targets, outputs.squeeze())
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # 2D image
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))

        # 3D image
        elif len(grads.shape) == 5:
            return np.mean(grads, axis=(2, 3, 4))

        else:
            raise ValueError("Invalid grads shape."
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")


class GradCAMPlusPlus(GradCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(GradCAMPlusPlus, self).__init__(model, target_layers,
                                              reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))
        return weights


def vis_cam(model, target_layers, net_input, canvas_img, gt_image):
    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=net_input, targets=gt_image)  # , targets = targets
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        # src_img = np.float32(canvas_img.transpose(2, 0, 1)) / 255
        src_img = np.float32(canvas_img) / 255
        visualization = show_cam_on_image(src_img, grayscale_cam, use_rgb=True)
        # cv2.imshow('feature map', visualization)
        # cv2.waitKey(0)
        # You can also get the model outputs without having to redo inference
        # model_outputs = cam.outputs
        return visualization
