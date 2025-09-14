import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    """ResNet50 modified to accept 55 input channels"""

    def __init__(self, num_classes=3, pretrained_backbone=False, in_channels=55):
        super(ResNet50, self).__init__()

        # Load standard ResNet50
        resnet50 = models.resnet50(pretrained=pretrained_backbone)

        # Modify the first convolutional layer to accept 55 channels
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # If using pretrained weights, initialize the new conv layer intelligently
        if pretrained_backbone:
            # Get the pretrained weights from the first layer
            pretrained_weights = resnet50.conv1.weight.data
            # Average the weights across RGB channels and repeat for 55 channels
            mean_weights = torch.mean(pretrained_weights, dim=1, keepdim=True)
            self.conv1.weight.data = mean_weights.repeat(1, 55, 1, 1) * (3.0 / 55.0)

        # Copy the rest of the ResNet50 architecture
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.avgpool = resnet50.avgpool
        self.fc = resnet50.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from torchsummary import summary

    model = ResNet50().cuda()
    x = torch.randn(1, 55, 256, 256)

    summary(model, (55, 256, 256))
