import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16

from ..super_resolution.utils import optical_flow_warp, L1_regularization

class ISeeBetterLoss(nn.Module):
    def __init__(self):
        super(ISeeBetterLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

def OFR_loss(x0, x1, optical_flow):
    warped = optical_flow_warp(x0, optical_flow)
    loss = torch.mean(torch.abs(x1 - warped)) + 0.1 * L1_regularization(optical_flow)
    return loss


class Module_CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=0.001):
        super(Module_CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, gt):
        return torch.mean(torch.sqrt((output - gt) ** 2 + self.epsilon ** 2))


# AdaCoF loss for Frame interpolation
class AdaCoF_loss(nn.modules.loss._Loss):
    def __init__(self, weight_charbonnier=1, weight_gSpatial=0.01, weight_gOcclusion=0.005):
        self.weight_charbonnier = weight_charbonnier
        self.weight_gSpatial = weight_gSpatial
        self.weight_gOcclusion = weight_gOcclusion

        self.charbonnier_loss = Module_CharbonnierLoss()

    def forward(self, output, gt, input_frames):
            losses = []

            # calculating charbonnerloss
            losses.append(self.weight_charbonnier *self.charbonnier_loss(output['frame1'], gt))
            # calculating gspatial component
            losses.append(self.weight_gSpatial * output['g_Spatial'])
            # calculating gocclusion component
            losses.append(self.weight_gOcclusion * output['g_Occlusion'])

            loss_sum = sum(losses)

            return loss_sum
