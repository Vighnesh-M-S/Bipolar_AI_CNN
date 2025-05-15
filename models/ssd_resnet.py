import torch
import torch.nn as nn
from torchvision.models import resnet50

class SSDResNet(nn.Module):
    def __init__(self, num_classes):
        super(SSDResNet, self).__init__()
        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool & fc

        self.class_head = nn.Conv2d(2048, num_classes * 3, kernel_size=3, padding=1)
        self.box_head = nn.Conv2d(2048, 4 * 3, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.backbone(x)
        class_preds = self.class_head(features)
        box_preds = self.box_head(features)

        # Reshape outputs
        class_preds = class_preds.permute(0, 2, 3, 1).reshape(x.size(0), -1, class_preds.size(1) // 3)
        box_preds = box_preds.permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        return class_preds, box_preds