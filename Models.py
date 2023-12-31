import torch.nn as nn
from torchvision import models

def resnet18():
    ''' ResNet 18 Model '''
    #model = models.resnet18(pretrained=True)
    model = models.resnet18(weights='DEFAULT')

    # To Freeze Layers
    for param in model.parameters():
        param.requires_grad = False

    # New Output Layers
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model



