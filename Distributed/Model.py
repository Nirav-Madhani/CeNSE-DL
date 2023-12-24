import torchvision.models as models
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input_channels=3, pretrained=True):
        super(CustomModel, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.resnet = nn.Sequential(*list(resnet.children())[1:-1])  # Remove first and last layer

        # Adjusting the first convolutional layer
        self.resnet[0] = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.fc = nn.Linear(resnet.fc.in_features, 3)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        return self.fc(x)