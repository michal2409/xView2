import numpy as np
import torch.nn as nn
import torchvision
import torch
    
class DeepLab(torch.nn.Module):
    def __init__(self, num_classes):
        super(DeepLab, self).__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):
        return self.model(x)['out']
