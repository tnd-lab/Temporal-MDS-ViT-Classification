import torch
import torch.nn as nn
import torchvision.models as models


class VGG16(nn.Module):
    """VGG16 modified to accept 55 input channels"""

    def __init__(self, num_classes=3, pretrained_backbone=False, in_channels=55):
        super(VGG16, self).__init__()

        # Load standard VGG16
        vgg16 = models.vgg16(pretrained=pretrained_backbone)

        # Modify the first convolutional layer to accept 55 channels
        # Original: Conv2d(3, 64, kernel_size=3, padding=1)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        # If using pretrained weights, initialize the new conv layer intelligently
        if pretrained_backbone:
            # Get the pretrained weights from the first layer
            pretrained_weights = vgg16.features[0].weight.data
            # Average the weights across RGB channels and repeat for 55 channels
            mean_weights = torch.mean(pretrained_weights, dim=1, keepdim=True)
            self.first_conv.weight.data = mean_weights.repeat(1, 55, 1, 1) * (
                3.0 / 55.0
            )
            self.first_conv.bias.data = vgg16.features[0].bias.data

        # Copy the rest of the VGG16 architecture
        self.features = nn.Sequential(
            self.first_conv,
            *list(vgg16.features.children())[1:],  # All layers except the first
        )

        self.avgpool = vgg16.avgpool
        self.classifier = vgg16.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    model = VGG16().cuda()
    x = torch.randn(2, 55, 256, 256)

    summary(model, (55, 256, 256))
