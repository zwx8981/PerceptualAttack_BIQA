import torch.nn as nn
from torchvision import models
from BCNN import BCNN
import torch

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

class BaseCNN(nn.Module):
    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.config = config
        if self.config.backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
        elif self.config.backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
        # elif self.config.backbone == 'resnet50':
        #     self.backbone = models.resnet50(pretrained=True)
        #     self.fc = nn.Linear(2048, 1)
        if config.std_modeling:
            outdim = 2
        else:
            outdim = 1
        if config.representation == 'BCNN':
            assert ((self.config.backbone == 'resnet18') | (self.config.backbone == 'resnet34')), "The backbone network must be resnet18 or resnet34"
            self.representation = BCNN()
            self.fc = nn.Linear(512 * 512, outdim)
        else:
            self.fc = nn.Linear(512, outdim)

        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # if self.config.fc:
        #     # Freeze all previous layers.
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False
        #     # Initialize the fc layers.
        #     nn.init.kaiming_normal_(self.fc.weight.data)
        #     if self.fc.bias is not None:
        #         nn.init.constant_(self.fc.bias.data, val=0)


    def forward(self, x):
        """Forward pass of the network.
        """
        x = self.norm(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        if self.config.representation == 'BCNN':
            x = self.representation(x)
        else:
            x = self.backbone.avgpool(x)

        x = self.fc(x)

        if self.config.std_modeling:
            mean = x[:, 0]
            t = x[:, 1]
            var = nn.functional.softplus(t)
            return mean, var
        else:
            return x
