from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights


def EfficientNetModelImport():
    model = efficientnet_b1(pretrained=True)

    return model


